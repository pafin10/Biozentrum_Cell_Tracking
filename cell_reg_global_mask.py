import numpy as np
import matplotlib.pyplot as plt
import h5py
from collections import defaultdict
import cv2

"""
This script serves to transform the CellReg output into a global mask that can be used to align the cells across sessions.
Objects of this class can be created in the evaluation script to generate the global mask.
The global mask is a list of contours, where each contour represents a cell's average footprint across all sessions.
All the methods used for the global mask from the other tested tool can then be applied to this one too. 
"""

class CellRegGlobalMask:
    def __init__(self, matfile):
        self.load_mat(matfile)
        self.convert_to_footprints(self.binary_footprints)
        self.average_footprints()
        self.get_avg_cell_contours()
        #self.reformat()
    
    def load_mat(self, matfile):
        with h5py.File(matfile, 'r') as f:
            struct = f['cell_registered_struct']
            self.centroids = struct['registered_cells_centroids'][:][:]
            self.centroids = [[self.centroids[0][i], self.centroids[1][i]] for i in range(len(self.centroids[0]))]
            self.mapping = struct['cell_to_index_map'][:]
            self.binary_footprints = struct['spatial_footprints_corrected'][:][0]
            self.n_sessions = len(self.binary_footprints)

            mapping = defaultdict(list)
            for i in range(self.n_sessions):
                if i > 0:
                    if self.mapping[0][i] > 0:
                        for j in range(len(self.mapping[i])):
                            mapping[j].append(int(self.mapping[i][j]) - 1) # -1 to make it 0-indexed, -1 now means no cell found

                if isinstance(self.binary_footprints[i], h5py.Reference):
                    target = struct[self.binary_footprints[i]]
                    assert isinstance(target, h5py.Dataset)
                    self.binary_footprints[i] = target[:]
                    self.binary_footprints[i] = np.transpose(self.binary_footprints[i], (2, 0, 1)) # (n_cells x 512 x 512)
            self.mapping = mapping
    
    def convert_to_footprints(self, binary_footprints):
        footprints = []
        for i in range(len(binary_footprints)):
            footprints.append([np.where(cell) for cell in binary_footprints[i]])
        self.footprints = footprints
    

    def average_footprints(self):
        # DEBUG THIS !!
        self.avg_footprints = []
        cnt = 0
        num_base_sessions = len(self.footprints[0])

        # Iterating over each cell and its corresponding list of indices
        for cell, cell_list in self.mapping.items():
            all_x, all_y = [], []

            # Make sure cnt does not exceed the footprint index range
            if cnt >= num_base_sessions:
                cnt = num_base_sessions - 1

            # Add the base session footprint to the lists
            base_x, base_y = self.footprints[0][cnt][0], self.footprints[0][cnt][1]
            all_x.extend(base_x)
            all_y.extend(base_y)

            counted = False
            valid_entries = 1  # Start counting with the base session

            # Iterate through each session in the cell list
            for k, index in enumerate(cell_list):
                if index != -1:
                    # Only increment cnt once per valid cell list if not counted
                    if not counted:
                        cnt += 1
                        counted = True

                    # Retrieve the additional session footprints
                    try:
                        cell_x, cell_y = self.footprints[k + 1][index]
                        all_x.extend(cell_x)
                        all_y.extend(cell_y)
                        valid_entries += 1
                    except IndexError:
                        # Catch index errors if footprints or cell indices are mismatched
                        print(f"Index error for cell {cell} at k={k}, index={index}")

            # Create numpy arrays for all collected coordinates
            all_x = np.array(all_x)
            all_y = np.array(all_y)

            # Define the size of the density map
            image_size = 512
            grid_size_x = image_size + 1
            grid_size_y = image_size + 1

            # Initialize the density map
            density_map = np.zeros((grid_size_y, grid_size_x))

            # Populate the density map with coordinates
            for x, y in zip(all_x, all_y):
                if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
                    density_map[y, x] += 1

            # Normalize the density map to create the average footprint
            normalized_footprint = density_map / valid_entries if valid_entries > 0 else density_map

            # Append the resulting normalized footprint to the average footprints list
            self.avg_footprints.append(normalized_footprint)

    def get_avg_cell_contours(self):
        self.global_mask = []
        global contour_threshold_ratio
        contour_threshold_ratio = 0.5

        for i in range(len(self.avg_footprints)):
            # Threshold the density map
            threshold = contour_threshold_ratio * np.max(self.avg_footprints[i])
            _, thresholded_map = cv2.threshold(self.avg_footprints[i], threshold, 1, cv2.THRESH_BINARY)

            # Get contours
            contours, _ = cv2.findContours(thresholded_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            # RETR_EXTERNAL retrieves only the outermost contours, 
            # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points
            contours = contours[0].tolist()
            contours.append(contours[0]) # Close the contour
            contours = [contours[i][0] for i in range(len(contours))]

            self.global_mask.append(contours)

    def reformat(self):
        new_global_mask = []
        for cell in self.global_mask:
            # Flatten the cell structure
            cell = cell[0]
            tmp = []
            for i in range(len(cell)): 
                point = cell[i]
                point = [point[0][0], point[0][1]]
                tmp.append(point)
            new_global_mask.append(tmp)
        self.global_mask = new_global_mask


if __name__ == '__main__':
    matfile = r'H:\Desktop\Code\Scores and Alignment\cell_reg\cellRegistered_20240819_103449.mat'
    cr = CellRegGlobalMask(matfile)
    print(cr.centroids.shape) # (2 x n_cells)
    for k,v in cr.mapping.items():
        print(k, v) # (n_sessions x n_cells)
    print(cr.footprints[0][0]) # (n_cells x 512 x 512)
    print(cr.avg_footprints)

        
    # Display the first cell footprint
    #plt.imshow(cr.footprints[0][1])
    #plt.show()


    