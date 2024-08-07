import numpy as np
import os 
import pickle
import matplotlib.pyplot as plt
from utils import *
from plotting import Plot
from collections import defaultdict
from shapes import Shapes
from scipy.stats import skew
from tqdm import tqdm
import time

# Constants
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
    pixels = []
    global_mask_cells = defaultdict(tuple)

    for i in range(len(filled_cells)):
        cell = filled_cells[i]
        y_pix_filled_cell = [int(round(p[0])) for p in cell]
        x_pix_filled_cell = [int(round(p[1])) for p in cell]
        filled_cells[i] = (y_pix_filled_cell, x_pix_filled_cell)
        global_mask_cells[i] = (y_pix_filled_cell, x_pix_filled_cell)


    for j in range(len(statsfiles)):
        start_time = time.time() 
        print("Session", j)
        same_cells = defaultdict(list)
        n = len(statsfiles[j])
        statsfile = statsfiles[j]
        session_cells = [(statsfile[i], i) for i in range(n) if cellfiles[j][i][0]]
        data = [cell[0] for cell in session_cells]
        y_pixels = [cell['ypix'] for cell in data]
        x_pixels = [cell['xpix'] for cell in data]
        all_pixels = [[(y, x) for x, y in zip(x_pix, y_pix)] for x_pix, y_pix in zip(x_pixels, y_pixels)]
        # calculate session centroids, more accurate than med
        session_centroids = compute_centroids(all_pixels)

        for k in range(len(session_cells)): 
            session_cell = session_cells[k][0]
            centroid1 = session_centroids[k][0] #session_cell['med']

            for i, (y_pix_filled_cell, x_pix_filled_cell) in enumerate(filled_cells):
                centroid2 = list(gm_centroids[i][0])
                if (dist(centroid1, centroid2) < distance_threshold):
                    overlap = calculate_overlap(session_cell['ypix'], session_cell['xpix'], y_pix_filled_cell, x_pix_filled_cell)
                    if overlap > 0.01:
                        same_cells[k].append((overlap, i))

            if same_cells[k]:
                overlap_tuples = same_cells[k]
                overlap_tuples.sort(key=lambda x: x[0], reverse=True)
                same_cells[k] = overlap_tuples

        same_cells = dict(sorted(same_cells.items(), key=lambda item: item[0]))
    
        #session_centroids[session_cell_ind] = session_cells[session_cell_ind][0]['med'] 

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
        
        # Align the centers of the cells from the session with the global mask
        global_mask_list_struct = convert_global_mask_to_list_structure(np_global_mask)
        session_cells_all_pixels_list_struct = [[[x, y] for x, y in zip(cell[0], cell[1])] for cell in flattened_session_cells.values()]
        session_cells_contours_list_struct = extract_contours(session_cells_all_pixels_list_struct)

        shapes = Shapes(global_mask_list_struct, session_cells_contours_list_struct)
        
        cells_w_aligned_centers = shapes.align_centers(gm_centroids, session_centroids)
        
        if j == 0:
            P.plot_cells_w_aligned_centers(cells_w_aligned_centers, "aligned_centers_0_", session_cell_idx=0)
                
        
        centered_cells_by_session = defaultdict(list)

        # TODO: do normalization per cell not globally, check how to prevent bias 
        dtw_distances = []
        for cell_pair in cells_w_aligned_centers:
            session_cell_coord = cell_pair[0][1]
            global_cell_coord = cell_pair[1][1]
            dtw_distance = shapes.DTW(session_cell_coord, global_cell_coord).calculate_shape_similarity()
            dtw_distances.append(dtw_distance)

        mini, maxi = min(dtw_distances), max(dtw_distances)
        mini_normalized, maxi_normalized = 0.0, 1.0

        # check skew and normalize DTW distances
        # values for skew and percentiles are quite arbitrary so check for more theoretical foundations
        P.plot_distribution(dtw_distances, "DTW_distance_distribution")
        skewness = skew(dtw_distances)
        if abs(skewness) > 1:
            print("The distribution of DTW distances is quite skewed.")
            if skewness < 1:
                mini = np.percentile(dtw_distances, 10)
                mini_normalized = 0.1
            else: 
                maxi = np.percentile(dtw_distances, 90)
                maxi_normalized = 0.9

        if maxi == mini:
            normalized_dtw_distances = [0.0] * len(dtw_distances)
        else:
            normalized_dtw_distances = [(distance - mini) / (maxi - mini) for distance in dtw_distances]

        # update the normalized distances
        # remember, session cell coordinates are centered, so differ for same index while global mask cell coordinates are the same
        for i in range(len(cells_w_aligned_centers)):
            cell_pair = cells_w_aligned_centers[i]
            idx_session_cell, session_cell_coord = cell_pair[0][0], cell_pair[0][1]
            idx_global_cell, global_cell_coord = cell_pair[1][0], cell_pair[1][1]
            dtw_normalized_distance = normalized_dtw_distances[i]
            centered_cells_by_session[idx_session_cell].append([idx_global_cell, session_cell_coord, global_cell_coord, dtw_normalized_distance])

        # sort by DTW distance and plot the warping path
        for v in centered_cells_by_session.values():
            v.sort(key=lambda x: x[-1])
        
        """
        The warping path is a sequence of index pairs (i,j) that defines the alignment between two sequences, 
        say A=(a1,a2,…,an) and B=(b1,b2,…,bm). The goal of DTW is to find this path such that the cumulative distance 
        between the aligned elements of the sequences is minimized.
        """

        # This plotting is very slow, if not needed, comment out
        """for k in range(len(centered_cells_by_session)):
            for i in range(len(centered_cells_by_session[k])):
                dtw = shapes.DTW(centered_cells_by_session[k][i][1], centered_cells_by_session[k][i][2])
                P.plot_dtw_results(dtw, title=f"Session 1 cell {k} vs global mask cell {centered_cells_by_session[k][i][0]}")
                """

        for session_cell, shape_cell_data in centered_cells_by_session.items():
            for session_cell1, overlap_cell_data in same_cells.items():
                found = False
                if session_cell == session_cell1:
                    for i in range(len(shape_cell_data)):
                        found = False
                        for m in range(len(overlap_cell_data)):
                            # i.e. it is the same gobal mask cell, so we exclude those that dont pass the overlap threshold
                            if overlap_cell_data[m][1] == shape_cell_data[i][0]:
                                dtw_distance = shape_cell_data[i][3]
                                updated_tuple = list(overlap_cell_data[m]) + [dtw_distance]
                                overlap_cell_data[m] = tuple(updated_tuple)
                                found = True
                        if not found:
                            gm_cell_index, dtw_distance = shape_cell_data[i][0], shape_cell_data[i][3]
                            overlap_cell_data.append((0, gm_cell_index, dtw_distance))


        # calculate weighted alignment score
        for session_cell, overlap_cell_data in same_cells.items():
            for i in range(len(overlap_cell_data)):
                overlap_cell = overlap_cell_data[i]
                if len(overlap_cell) == 3:
                    overlap = overlap_cell[0]
                    dtw = overlap_cell[2]
                    alignment_score = compute_cell_alignment_score(overlap=overlap, dtw_distance=dtw, dtw_max=maxi_normalized, weights=[0.5, 0.5])
                    overlap_cell = overlap_cell + (alignment_score,)
                    overlap_cell_data[i] = overlap_cell  # Update the tuple in the list
            overlap_cell_data.sort(key=lambda x: x[-1], reverse=True)
            if len(overlap_cell_data) > 0 and isinstance(overlap_cell_data[0][-1], float):
                print("The best alignment for cell", session_cell, "is with cell", overlap_cell_data[0][1], "from the global mask with alignment score", overlap_cell_data[0][-1])
                

        # now check visually by plotting if alignment score seems to correctly align cells        
        for session_cell, overlap_cell_data in same_cells.items():
            for overlap_cell in overlap_cell_data:
                if len(overlap_cell) == 4:
                    P.plot_mask("global_mask", idx=overlap_cell[1], session_cell=session_cell, session_pixels=flattened_session_cells, 
                                overlap_score=overlap_cell[0], session=sessions[j], alignment_score=overlap_cell[3])
                    
        print("One session takes", time.time() - start_time, "seconds to run")

            
        