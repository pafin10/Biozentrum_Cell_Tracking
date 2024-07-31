import numpy as np
import os 
import random
import pickle
import matplotlib.pyplot as plt
from utils import *
from plotting import Plot
from collections import defaultdict
from shapes import Shapes


global distance_threshold
global sample_size
global image_shape
distance_threshold = 40
sample_size = 0.4
image_shape = (512, 512)



if __name__ == '__main__':
    # paths
    root = r'H:\Desktop\Code\DON-019539_B'
    statsfiles, opsfiles, cellfiles = load_files(root)
    global_mask_file_path = r'\\biopz-jumbo.storage.p.unibas.ch\rg-fd02$\_Members\fingl0000\Desktop\Code\DON-019539_B\master_mask_GUI.pkl'
    session_mask_file_path = r'\\biopz-jumbo.storage.p.unibas.ch\rg-fd02$\_Members\fingl0000\Desktop\Code\DON-019539_B\session_mask_GUI.pkl'
    alignment_npz = r'\\biopz-jumbo.storage.p.unibas.ch\rg-fd02$\_Members\fingl0000\Desktop\Code\DON-019539_B\20240523\alignment\alignment_parameters.npz'
    output_dir = r'D:\Outputs\Alignment_DON-019539'
    recording_years = ['2023', '2024']
    sessions = [dir for dir in os.listdir(root) if any(dir.startswith(year) for year in recording_years)]

    # Load masks
    with open(global_mask_file_path, 'rb') as file:
        global_mask = pickle.load(file)
    with open(session_mask_file_path, 'rb') as file:
        session_mask = pickle.load(file)

    # Load the alignment parameters
    alignment_parameters = np.load(alignment_npz)
    for file in alignment_parameters.files:
        print(file, alignment_parameters[file])

    # Convert the global mask to a NumPy array with dtype=object and convert to binary mask
    np_global_mask = np.array(global_mask, dtype=object)
    np_session_mask = np.array(session_mask, dtype=object)
    binary_global_mask = convert_to_binary_mask(np_global_mask, image_shape)

    # Fill the area of each cell in the global mask with pixels to compute overlap and plot the filled cells
    print("Filling cell areas in global mask...")
    m, filled_cells = fill_cells(np_global_mask, image_shape)
    P = Plot(np_global_mask, binary_global_mask, os.path.join(output_dir, "plots"))
    gm_centroids = compute_centroids(filled_cells)
    P.plot_filled_cells(m, centroids=gm_centroids.values())


    # compute overlap between cells from each session with global mask
    same_cells = defaultdict(list)
    pixels = []
    global_mask_cells = defaultdict(tuple)

    for i in range(len(filled_cells)):
        cell = filled_cells[i]
        y_pix_filled_cell = [int(round(p[0])) for p in cell]
        x_pix_filled_cell = [int(round(p[1])) for p in cell]
        global_mask_cells[i] = (y_pix_filled_cell, x_pix_filled_cell)


    for j in range(len(statsfiles)): 
        print("Session", j)
        same_cells = defaultdict(list)
        n = len(statsfiles[j])
        statsfile = statsfiles[j]
        session_cells = [(statsfile[i], i) for i in range(n) if cellfiles[j][i][0]]

        for k in range(len(session_cells)): 
            session_cell = session_cells[k][0]
            for i in range(len(filled_cells)):
                gm_cell = filled_cells[i]
                centroid1 = session_cell['med']
                centroid2 = list(gm_centroids[i][0])
                if (dist(centroid1, centroid2) < distance_threshold):
                    y_pix_filled_cell = [int(round(p[0])) for p in gm_cell]
                    x_pix_filled_cell = [int(round(p[1])) for p in gm_cell]
                    overlap = calculate_overlap(session_cell['ypix'], session_cell['xpix'], y_pix_filled_cell, x_pix_filled_cell)
                    if overlap > 0.01:
                        same_cells[k].append((overlap, i))

            if same_cells[k]:
                overlap_tuples = same_cells[k]
                overlap_tuples.sort(key=lambda x: x[0], reverse=True)
                same_cells[k] = overlap_tuples

        same_cells = dict(sorted(same_cells.items(), key=lambda item: item[0]))

        session_centroids = [None] * len(session_cells)
        for orig_cell_ind in same_cells.keys():
            session_centroids[orig_cell_ind] = session_cells[orig_cell_ind][0]['med'] 

        print("There are", len(same_cells), "cells with overlap > 0.01")
        for cell, corresponding_cell in same_cells.items():
            if len(corresponding_cell) > 0:
                print("Cell", cell, "from the original recording corresponds to cell", corresponding_cell[0][1], "from the global mask with overlap", corresponding_cell[0][0])
                print("There were", len(corresponding_cell), "cells with overlap > 0.01")

        session_cells_pixels = {i: (session_cells[i][0]['ypix'], session_cells[i][0]['xpix']) for i in range(len(session_cells))}   
        flattened_session_cells = {
            key: (list(value[0]), list(value[1]))
            for key, value in session_cells_pixels.items()
        }
        
        P.plot_cells_with_overlap(same_cells, flattened_session_cells, global_mask_cells)

        # Plot the overlap distribution for each overlapping cell 
        """
         if j == 1:
            for session_cell, overlap_cells in same_cells.items():
                if overlap_cells:
                    P.plot_overlap_distribution(session_cell, overlap_cells, output_dir)
                    if len(overlap_cells) > 1 and overlap_cells[1][0] * 2 > overlap_cells[0][0]:
                        print("Cell", session_cell, "'s overlap with cell", overlap_cells[0][1], "from the global mask is more than 0.5 of maximum overlap")
                    for overlap_cell in overlap_cells:
                        P.plot_mask("global_mask", idx=overlap_cell[1], session_cell=session_cell, session_pixels=flattened_session_cells, 
                                    overlap_score=overlap_cell[0], session=sessions[j])
        
        """
       
        
        
        # TODO: Implement the align_centers method in the Shapes class, then find a way to quantify shape similarity. Create a weighted average of the overlap and shape similarity scores.
        
        global_mask_list_struct = convert_global_mask_to_list_structure(np_global_mask)
        session_cells_list_struct = [[[x, y] for x, y in zip(cell[0], cell[1])] for cell in flattened_session_cells.values()]
        shapes = Shapes(global_mask_list_struct, session_cells_list_struct)
        
        cells_w_aligned_centers = shapes.align_centers(gm_centroids, session_centroids)
        if j == 1:
            P.plot_cells_w_aligned_centers(cells_w_aligned_centers, "aligned_centers")
        
        centered_cells_by_session = defaultdict(list)

        for cell_pair in cells_w_aligned_centers:
            idx_session_cell, session_cell_coord = cell_pair[0][0], cell_pair[0][1]
            idx_global_cell, global_cell_coord = cell_pair[1][0], cell_pair[1][1]
            centered_cells_by_session[idx_session_cell].append([idx_global_cell, session_cell_coord, global_cell_coord])
            shape_similarity = shapes.calculate_shape_similarity(session_cell_coord, global_cell_coord)
            centered_cells_by_session[idx_session_cell].append(shape_similarity)
        
        # TODO: Debug issue with unhashable type dict above

  

        for session_cell in centered_cells_by_session.keys():
            print(centered_cells_by_session[session_cell])
       
        for session_cell in centered_cells_by_session.keys():
            centered_cells_by_session[session_cell].sort(key=lambda x: x[-1])
            print("Session cell", session_cell, "has the following aligned centers, index: {idx} with shape similarity:".format(centered_cells_by_session[session_cell][0]), 
                  centered_cells_by_session[session_cell][-1])
       