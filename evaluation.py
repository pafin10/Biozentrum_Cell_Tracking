import numpy as np
import os 
import pickle
import matplotlib.pyplot as plt
from utils import *
from plotting import Plot
from collections import defaultdict
from shapes import Shapes
from scipy.stats import skew
import time
from cell_reg_global_mask import CellRegGlobalMask
from scipy.interpolate import interp1d
import tkinter as tk
from PIL import Image, ImageTk
from GUI import QuestionPrompt, ManualCorrection

""""
Most plotting has been commented out as it is not necessary for the evaluation of the alignment and takes up the majority of the script's runtime.
If you want to plot the cells, just uncomment the relevant lines 
(they make use of a plotting object "P" and the functions are named accordingly).
If you dont want to be prompted about manual correction, comment out line 390.
"""


# Constants
distance_threshold = 40
sample_size = 0.4
image_shape = (512, 512)
overlap_threshold = 0.01
n_cells_dtw_threshold = 2 # minimum number of cells to use DTW, else the centered overlap is used instead
manual_threshold = None


def print_overlap_info(same_cells, overlap_threshold):
    print("There are", len(same_cells), "cells with overlap > {}".format(overlap_threshold))
    for cell, corresponding_cell in same_cells.items():
        if len(corresponding_cell) > 0:
            print("Cell", cell, "from the original recording corresponds to cell", corresponding_cell[0][1], 
                    "from the global mask with overlap", corresponding_cell[0][0])
            print("There were", len(corresponding_cell), "cells with overlap > {}".format(overlap_threshold))

def find_overlaps(same_cells, session_cells, filled_cells, distance_threshold, overlap_threshold):
    for k in range(len(session_cells)): 
            session_cell = session_cells[k][0]
            centroid1 = session_centroids[k][0] 

            for i, (y_pix_filled_cell, x_pix_filled_cell) in enumerate(filled_cells):
                #centroid2 = list(cell_reg_centroids[i][0])
                centroid2 = list(gm_centroids[i][0])
                if (dist(centroid1, centroid2) < distance_threshold):
                    overlap = calculate_overlap(session_cell['ypix'], session_cell['xpix'], y_pix_filled_cell, x_pix_filled_cell)
                    if overlap > overlap_threshold:
                        same_cells[k].append((overlap, i))

            if same_cells[k]:
                overlap_tuples = same_cells[k]
                overlap_tuples.sort(key=lambda x: x[0], reverse=True)
                same_cells[k] = overlap_tuples
    
    same_cells = dict(sorted(same_cells.items(), key=lambda item: item[0]))

def downsample_sequence(seq, target_length):
    """
    Downsample a sequence to match the target length by linear interpolation.
    
    :param seq: The sequence to downsample (list or numpy array).
    :param target_length: The length to which the sequence should be downsampled.
    :return: The downsampled sequence.
    """
    seq = np.array(seq)
    original_length = len(seq)

    if original_length == target_length:
        return seq

    # Extract x and y coordinates
    x_coords = seq[:, 0]
    y_coords = seq[:, 1]

    # Original indices
    original_indices = np.linspace(0, original_length - 1, original_length)
    new_indices = np.linspace(0, original_length - 1, target_length)

    # Interpolation functions
    interp_x = interp1d(original_indices, x_coords, kind='linear', fill_value='extrapolate')
    interp_y = interp1d(original_indices, y_coords, kind='linear', fill_value='extrapolate')

    # Interpolate
    downsampled_x = interp_x(new_indices)
    downsampled_y = interp_y(new_indices)

    # Combine x and y coordinates into pairs
    downsampled_seq = np.vstack((downsampled_x, downsampled_y)).T

    return downsampled_seq

def compute_dtw_distances(cells_w_aligned_centers, dtw_distances, dtw_distances_list):
    for cell_pair in cells_w_aligned_centers:
        session_cell = cell_pair[0][0]
        global_cell = cell_pair[1][0]
        session_cell_coord = cell_pair[0][1]
        global_cell_coord = cell_pair[1][1]
         
        len_session = len(session_cell_coord)
        len_global = len(global_cell_coord)
        
        if len_session > len_global:
            # Downsample session_cell_coord to the length of global_cell_coord
            session_cell_coord_downsampled = downsample_sequence(session_cell_coord, len_global)
            sequence = session_cell_coord_downsampled
            global_cell_coord_downsampled = global_cell_coord  # No downsampling needed for global_cell_coord
        else:
            # Downsample global_cell_coord to the length of session_cell_coord
            global_cell_coord_downsampled = downsample_sequence(global_cell_coord, len_session)
            sequence = session_cell_coord
            # No downsampling needed for session_cell_coord
        
        

        # Compute DTW distance and path
        dtw_distance = np.inf
        dtw_path = None

        # just to test if better wo downsampling
        #sequence = session_cell_coord
        #global_cell_coord_downsampled = global_cell_coord
        # Find the closest point in the global cell to the starting point of the session cell to force starting point

        start = sequence[0]
        min_dist = np.inf
        point_index = None
        for i, point in enumerate(global_cell_coord_downsampled):
            dist = euclidean_distance(start, point)  
            if dist < min_dist:
                min_dist = dist
                point_index = i  

        # transformations to fix starting points and create sensible dtw alignment
        global_cell_coord_downsampled = np.concatenate(
            (global_cell_coord_downsampled[point_index+1 : ], global_cell_coord_downsampled[:point_index])
            )
        global_cell_coord_downsampled = np.delete(global_cell_coord_downsampled, 0, axis=0)
        global_cell_coord_downsampled = global_cell_coord_downsampled[::-1]

        # lengths must match, for now quick fix by removing the middle points
        mid = len(sequence) // 2
        np.delete(sequence, mid, 0)
        np.delete(sequence, mid, 0)

        permutations = generate_cyclic_permutations(sequence)
        for perm in permutations:
            dtw_instance = shapes.DTW(perm, global_cell_coord_downsampled)
            dtw_distance = min(dtw_instance.dtw_dist, dtw_distance)
            if dtw_distance == dtw_instance.dtw_dist:
                dtw_path = dtw_instance.dtw_path
        
        dtw_distances[session_cell].append((dtw_distance, global_cell))
        P.plot_dtw_alignment(sequence, global_cell_coord_downsampled, dtw_path, session_cell_idx=session_cell, global_cell_idx=global_cell)
        dtw_distances_list.append(dtw_distance)

def normalize_dtw_distances(cells_w_aligned_centers, gm_cells_footprints, dtw_distances):
    for session_cell in dtw_distances.keys():
        distances = dtw_distances[session_cell]
        mean = np.mean(distances, axis=0)[0]                
        std = np.std(distances)
        z_scores = [(distance[0] - mean) / std for distance in distances]
        if len(z_scores) > 2:  # Avoid division by zero
            min_z, max_z = min(z_scores), max(z_scores)
            normalized_z_scores = [(z - min_z) / (max_z - min_z) for z in z_scores]
        else:
            normalized_z_scores = [0.5] * len(z_scores)  # If there's only one distance, set it to 0.5 (mid-range)
        
        # we also use centered overlap scores and later weigh them differently according to number of potential candidates
        centered_overlap_scores = []
        gm_cell_xpixel, gm_cell_ypixel = [], []

        for t in range(len(distances)):
            gm_cell_index = distances[t][1]
            gm_cell_xpixel.append(gm_cells_footprints[gm_cell_index][0])
            gm_cell_ypixel.append(gm_cells_footprints[gm_cell_index][1])

        g = session_cell
        # find the index of the session cell in the list of cells with aligned centers
        while session_cell != cells_w_aligned_centers[g][0][0]:
            g += 1
    
        for t in range(len(gm_cell_xpixel)):
            sess_cell_contours = cells_w_aligned_centers[g][0][1]
            sess_cell_x_pix_contours = [x for x, y in sess_cell_contours]
            sess_cell_y_pix_contours = [y for x, y in sess_cell_contours]
            contours_sess_cell = [sess_cell_x_pix_contours, sess_cell_y_pix_contours]
            sess_cell_x_pix, sess_cell_y_pix = fill_cells(contours_sess_cell, image_shape)
            centered_overlap_scores.append(calculate_overlap(sess_cell_y_pix, sess_cell_x_pix, gm_cell_ypixel[t], gm_cell_xpixel[t]))
            g += 1

        dtw_distances[session_cell] = (normalized_z_scores, centered_overlap_scores)

def store_cell_pairs_with_scores(centered_cells_by_session, cells_w_aligned_centers, dtw_distances):
    last = -1
    for i in range(len(cells_w_aligned_centers)):
        cell_pair = cells_w_aligned_centers[i]
        idx_session_cell, session_cell_coord = cell_pair[0][0], cell_pair[0][1]
        idx_global_cell, global_cell_coord = cell_pair[1][0], cell_pair[1][1]
        if last != idx_session_cell:
            s = 0
        else:
            s += 1
        dtw_normalized_distance = dtw_distances[idx_session_cell][0][s]
        centered_overlap_score = dtw_distances[idx_session_cell][1][s]
        centered_cells_by_session[idx_session_cell].append([idx_global_cell, session_cell_coord, global_cell_coord, 
                                                            dtw_normalized_distance, centered_overlap_score])
        last = idx_session_cell

    # sort by DTW distance
    for v in centered_cells_by_session.values():
        v.sort(key=lambda x: x[-1])

def store_cell_pairs_with_all_scores(centered_cells_by_session, same_cells):
    for session_cell, shape_cell_data in centered_cells_by_session.items():
        for session_cell1, overlap_cell_data in same_cells.items():
            found = False
            if session_cell == session_cell1:
                for i in range(len(shape_cell_data)):
                    found = False
                    for m in range(len(overlap_cell_data)):
                        # i.e. it is the same gobal mask cell
                        if overlap_cell_data[m][1] == shape_cell_data[i][0]:
                            dtw_distance = shape_cell_data[i][3]
                            centered_overlap_score = shape_cell_data[i][4]
                            updated_tuple = list(overlap_cell_data[m]) + [dtw_distance] + [centered_overlap_score]
                            overlap_cell_data[m] = tuple(updated_tuple)
                            found = True
                    if not found: # if no overlap but still inside the chosen radius, add the cell to the list
                        gm_cell_index, dtw_distance, centered_overlap_score = shape_cell_data[i][0], shape_cell_data[i][3], shape_cell_data[i][4]
                        overlap_cell_data.append((0, gm_cell_index, dtw_distance, centered_overlap_score))
     
def compute_weighted_alignment_scores(same_cells, n_cells_dtw_threshold, aligned_cells):
    for session_cell, overlap_cell_data in same_cells.items():
            using_dtw = True
            n_cells = len(overlap_cell_data)
            if n_cells < n_cells_dtw_threshold:
                using_dtw = False
            for i in range(n_cells):
                overlap_cell = overlap_cell_data[i]
                if len(overlap_cell) == 4:
                    overlap = overlap_cell[0]
                    dtw = overlap_cell[2]
                    centered_overlap_score = overlap_cell[3]
                    alignment_score = compute_cell_alignment_score(overlap=overlap, dtw_distance=dtw, centered_overlap_score=centered_overlap_score, 
                                                                   n_cells=n_cells, using_dtw=using_dtw)
                    overlap_cell = overlap_cell + (alignment_score,)
                    overlap_cell_data[i] = overlap_cell  # Update the tuple in the list
            overlap_cell_data.sort(key=lambda x: x[-1], reverse=True)
            
            if len(overlap_cell_data) > 0 and isinstance(overlap_cell_data[0][-1], float):
                print("The best alignment for cell", session_cell, "is with cell", overlap_cell_data[0][1], 
                      "from the global mask with match score", overlap_cell_data[0][-1])
                aligned_cells += 1
    
    return aligned_cells

def map_best_cell_pairs(same_cells, cells_w_aligned_centers, matched_cells):
    for cell_pair in cells_w_aligned_centers:
            gm_index = cell_pair[1][0]
            for session_cell, overlap_cell_data in same_cells.items():
                while len(overlap_cell_data) > 0 and len(overlap_cell_data[0]) < 4:
                    overlap_cell_data.pop(0)
                if len(overlap_cell_data) > 0:
                    overlap_cell = overlap_cell_data[0]
                    if overlap_cell[1] == gm_index and cell_pair[0][0] == session_cell:
                        matched_cells.append(cell_pair)
                        break

def manual_correction_gui(session_cell, overlap_cell, cells_w_aligned_centers, matched_cells): 
    paths = []
    for overlap_cell in overlap_cells:  
        P.plot_mask("Global mask", idx=overlap_cell[1], session_cell=session_cell, session_pixels=flattened_session_cells, 
                                overlap_score=overlap_cell[0], session=sessions[current_session], alignment_score=overlap_cell[-1])
        paths.append(P.cell_pair_path)
    
    root = tk.Tk()
    root.title("Cell Alignment")

    photos, labels = [], []
    gridrow, gridcol = 0, 0

    for i in range(len(paths)):
        path = paths[i]
        gm_cell_idx = overlap_cells[i][1]
        image = Image.open(path)
        curr_size = image.size
        new_size_h = curr_size[0] 
        new_size_w = curr_size[1] 
        image = image.resize((new_size_h, new_size_w))
        photo = ImageTk.PhotoImage(image)
        photos.append(photo)
        labels.append(tk.Label(root, image=photo, text="{}".format(gm_cell_idx)))
        labels[i].grid(row=gridrow, column=gridcol)
        gridcol += 1
        if gridcol == 2:
            gridrow += 1
            gridcol = 0
    
    manual_correction_instance = ManualCorrection(root, labels, cells_w_aligned_centers, matched_cells, session_cell)
    
    root.mainloop()

def start_widget():
    prompt = QuestionPrompt()
    if prompt.get_result() == "Yes":
        manual_threshold = prompt.threshold
        while not manual_threshold.isdigit():
            print("Please enter a valid number.")
            manual_threshold = QuestionPrompt().threshold
    else:
        manual_threshold = None
    return manual_threshold
   
   

if __name__ == '__main__':
   
    # paths
    root_dir = r'H:\Desktop\Code\DON-019539_B'
    statsfiles, opsfiles, cellfiles = load_files(root_dir)
    global_mask_file_path = r'\\biopz-jumbo.storage.p.unibas.ch\rg-fd02$\_Members\fingl0000\Desktop\Code\DON-019539_B\master_mask_GUI.pkl'
    session_mask_file_path = r'\\biopz-jumbo.storage.p.unibas.ch\rg-fd02$\_Members\fingl0000\Desktop\Code\DON-019539_B\session_mask_GUI.pkl'
    alignment_npz = r'\\biopz-jumbo.storage.p.unibas.ch\rg-fd02$\_Members\fingl0000\Desktop\Code\DON-019539_B\20240523\alignment\alignment_parameters.npz'
    output_dir = r'D:\Outputs\Alignment_DON-019539'
    recording_years = ['2023', '2024']
    sessions = [dir for dir in os.listdir(root_dir) if any(dir.startswith(year) for year in recording_years)]
    matfile = r'H:\Desktop\Code\Scores and Alignment\cell_reg\cellRegistered_20240819_103449.mat'

    # Load masks
    with open(global_mask_file_path, 'rb') as file:
        global_mask = pickle.load(file)
    with open(session_mask_file_path, 'rb') as file:
        session_mask = pickle.load(file)

    # Load the alignment parameters
    alignment_parameters = np.load(alignment_npz)
    print("Alignment parameters have been loaded")

    # Convert the global mask to a NumPy array with dtype=object and convert to binary mask
    np_global_mask = np.array(global_mask, dtype=object)
    np_session_mask = np.array(session_mask, dtype=object)
    binary_global_mask = convert_to_binary_mask(np_global_mask, image_shape)

    # Load CellRegGlobalMask object
    cell_reg = CellRegGlobalMask(matfile)
    cell_reg_gm = cell_reg.global_mask
    cell_reg_centroids = cell_reg.centroids
    cell_reg_filled_cells = fill_cells_gm(cell_reg_gm, image_shape)
    """
    tmp = cell_reg.footprints
    
    #cell_reg_filled_cells = list(cell_reg_filled_cells.values())
    cell_reg_centroids = compute_centroids(cell_reg_filled_cells)

    cnt = 0
    for i in range(len(tmp)):
        for j in range(len(tmp[i])):

            x_coords = tmp[i][j][0]
            y_coords = tmp[i][j][1]

            cell_reg_filled_cells[i][j] = [[x_coords[k], y_coords[k]] for k in range(len(x_coords))]
            cnt += 1
    """
    

    # Fill the area of each cell in the global mask with pixels to compute overlap and plot the filled cells
    print("Filling cell areas in global mask...")
    m, gm_cells_footprints = fill_cells_gm(np_global_mask, image_shape)
    #binary_cell_reg_mask = convert_to_binary_mask(cell_reg_gm, image_shape)
    #P = Plot(cell_reg_gm, binary_cell_reg_mask, os.path.join(output_dir, "plots_cellreg"))

    # Initialize Plot object with global mask and fill cells
    P = Plot(np_global_mask, binary_global_mask, os.path.join(output_dir, "plots"))
    gm_centroids = compute_centroids(gm_cells_footprints)
    P.plot_filled_cells(m, centroids=gm_centroids.values())


    # transform the filled cells to pixel coordinates
    for i in range(len(gm_cells_footprints)):
        cell = gm_cells_footprints[i]
        y_pix_filled_cell = [int(round(p[0])) for p in cell]
        x_pix_filled_cell = [int(round(p[1])) for p in cell]
        gm_cells_footprints[i] = (y_pix_filled_cell, x_pix_filled_cell)



    cell_mappings = []
    manual_threshold = start_widget()
    if manual_threshold:
        print("The manual threshold for close cases has been set to", manual_threshold)
    else:
        print("No manual threshold has been set for close cases.")


    ############################### Cell Alignment ########################################

    for current_session in range(len(statsfiles)):
        if current_session == 0:
            continue


        start_time = time.time() 
        print("Session", current_session)
        
        same_cells = defaultdict(list)
        n = len(statsfiles[current_session])
        statsfile = statsfiles[current_session]
        session_cells = [(statsfile[i], i) for i in range(n) if cellfiles[current_session][i][0]]
        tmp = [cell[0] for cell in session_cells]
        y_pixels = [cell['ypix'] for cell in tmp]
        x_pixels = [cell['xpix'] for cell in tmp]

        all_pixels = [[(y, x) for x, y in zip(x_pix, y_pix)] for x_pix, y_pix in zip(x_pixels, y_pixels)]
        # calculate session centroids, more accurate than med, important for metrics
        session_centroids = compute_centroids(all_pixels)

        find_overlaps(same_cells, session_cells, gm_cells_footprints, distance_threshold, overlap_threshold)
        print_overlap_info(same_cells, overlap_threshold)

        # store the pixel coordinates of the session cells for plotting
        session_cells_pixels = {i: (session_cells[i][0]['ypix'], session_cells[i][0]['xpix']) for i in range(len(session_cells))}   
        flattened_session_cells = {key: (list(value[0]), list(value[1])) for key, value in session_cells_pixels.items()}

        # plot all cells with overlap and their overlap distribution
        P.plot_distribution_of_overlap_number(same_cells, session=current_session)
        P.plot_cells_with_overlap(same_cells, flattened_session_cells, filled_global_cells=gm_cells_footprints)

        # Plot the overlap distribution for each overlapping cell 
        """
        for session_cell, overlap_cells in same_cells.items():
            if overlap_cells:
                P.plot_overlap_distribution(session_cell, overlap_cells, output_dir)
                for overlap_cell in overlap_cells:
                    P.plot_mask("global_mask", idx=overlap_cell[1], session_cell=session_cell, session_pixels=flattened_session_cells, 
                                overlap_score=overlap_cell[0], session=current_session)
        """
        
        # Align the centers of the cells from the session with the global mask
        global_mask_list_struct = convert_global_mask_to_list_structure(np_global_mask) 
        #global_mask_list_struct = cell_reg_gm
        session_cells_all_pixels_list_struct = [[[x, y] for x, y in zip(cell[0], cell[1])] for cell in flattened_session_cells.values()]
        session_cells_contours_list_struct = extract_contours(session_cells_all_pixels_list_struct)

        # Initialize Shapes object and align the centers of the cells on a per - cell basis
        shapes = Shapes(global_mask_list_struct, session_cells_contours_list_struct)
        cells_w_aligned_centers = shapes.align_centers(gm_centroids, session_centroids)
        #cells_w_aligned_centers = shapes.align_centers(cell_reg_centroids, session_centroids)

        # plot example cells
        """
        for m in range(len(session_cells)):
            P.plot_cells_w_aligned_centers(cells_w_aligned_centers, title="Aligned centers session {} cell {}".
                                           format(sessions[current_session], m), session_cell=m, session=sessions[current_session])
                                           
        P.plot_cells_w_aligned_centers(cells_w_aligned_centers, "Aligned centers for session {}".format(sessions[current_session]))
        """

        # Compute dtw distance between session and global mask cells
        # DTW is not capable of considering modular wrapping over the borders of arrays and it maintains monotonicity 
        # As centered cell coordinates do not have the same starting points and also vary in length, we iterate through permutations of 
        # the coordinate array to find the arrangement that minimizes the dtw distance
        dtw_distances = defaultdict(list)
        dtw_distances_list = []
        compute_dtw_distances(cells_w_aligned_centers, dtw_distances, dtw_distances_list)

        # Normalize DTW distances per session cell
        normalize_dtw_distances(cells_w_aligned_centers, gm_cells_footprints, dtw_distances)
        
        # check skew 
        P.plot_distribution(dtw_distances_list, "Distribution of DTW distances")
        skewness = skew(dtw_distances_list)
        if abs(skewness) > 1:
            print("The distribution of DTW distances is quite skewed.")
        print("The skewness of the distribution of DTW distances is", skewness)

        # store the normalized DTW distances and centered overlap scores in the same data structure
        # remember, session cell coordinates are shifted by the distance to the centroid of the paired global mask cell, 
        # so they differ for the same index while global mask cell coordinates are unchanged for a given index. 
        centered_cells_by_session = defaultdict(list)
        store_cell_pairs_with_scores(centered_cells_by_session, cells_w_aligned_centers, dtw_distances)

        # store overlap and dtw distance in one data structure with cell indices
        store_cell_pairs_with_all_scores(centered_cells_by_session, same_cells)
        
        # calculate weighted alignment score
        aligned_cells = 0
        aligned_cells = compute_weighted_alignment_scores(same_cells, n_cells_dtw_threshold, aligned_cells)
        
        print("Out of {} session cells, {} cells were aligned.".format(len(session_cells), aligned_cells))
                
        # store max score global mask cells for each session cell with coordinates to plot whole field of view
        matched_cells = []
        map_best_cell_pairs(same_cells, cells_w_aligned_centers, matched_cells)
        
        #P.plot_cells_w_aligned_centers(matched_cells, "Best alignment for session {}".format(sessions[current_session]))

        # now check visually by plotting if match score seems to correctly align cells 
        last = 0    
        overlap_cells = []             
        plot_gui = False
  
        for session_cell, overlap_cell_data in same_cells.items():
            for overlap_cell in overlap_cell_data:
                if len(overlap_cell) == 5:
                    if manual_threshold:
                        if session_cell != last:
                            if plot_gui:
                                manual_correction_gui(last, overlap_cells, cells_w_aligned_centers, matched_cells)
                                overlap_cells = [overlap_cell]
                                plot_gui = False
                            else: 
                                overlap_cells = [overlap_cell]
                        else:
                            overlap_cells.append(overlap_cell)
                        if len(overlap_cell_data) > 1 and overlap_cell_data[1][-1] > overlap_cell_data[0][-1] * int(manual_threshold) / 100:
                            plot_gui = True 
                        last = session_cell
                    
                    P.plot_mask("Global mask", idx=overlap_cell[1], session_cell=session_cell, session_pixels=flattened_session_cells, 
                            overlap_score=overlap_cell[0], session=sessions[current_session], alignment_score=overlap_cell[-1])
                   
        cell_mappings.append(same_cells)            
        print("One session takes", time.time() - start_time, "seconds to run")

    # Structure of same_cells: {session_cell: [i, j, k...]} where i = (overlap, global_cell_idx, dtw_distance, centered_overlap, match_score)  
