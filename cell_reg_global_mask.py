import numpy as np
import matplotlib.pyplot as plt
import h5py
from collections import defaultdict
import cv2

class CellRegGlobalMask:
    def __init__(self, matfile):
        self.load_mat(matfile)
        self.footprints = self.convert_to_footprints(self.binary_footprints)
        self.average_footprints()
        self.get_avg_cell_contours()
        self.reformat()
    
    def load_mat(self, matfile):
        with h5py.File(matfile, 'r') as f:
            print("keys: %s" % f.keys())
            struct = f['cell_registered_struct']
            self.centroids = struct['registered_cells_centroids'][:]
            self.mapping = struct['cell_to_index_map'][:]
            self.binary_footprints = struct['spatial_footprints_corrected'][:][0]
            self.n_sessions = len(self.binary_footprints)

            mapping = defaultdict(list)
            l = len(self.mapping[0])
            for i in range(self.n_sessions):
                if i > 0:
                    if self.mapping[0][i] > 0:
                        for j in range(len(self.mapping[i])):
                            mapping[j].append(int(self.mapping[i][j]) - 1) # -1 to make it 0-indexed, -1 now means no cell found
                    """else:
                    cnt = 0
                    if i > 0:
                        for j in range(len(self.mapping[i])):
                            mapping[l + cnt].append(-1)
                            cnt += 1"""

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
        return footprints
    

    def average_footprints(self):
        self.avg_footprints = []
        cnt = 0
        # Assuming each cell's footprint is stored as a list of (x, y) coordinates
        for cell, cell_list in self.mapping.items():
            all_x, all_y = [], []
            if cnt == len(self.footprints[0]) - 1:
                break
            # Start with the base session
            base_x, base_y = self.footprints[0][cnt][0], self.footprints[0][cnt][1]
            all_x.extend(base_x)
            all_y.extend(base_y)

            counted = False
            # Add points from other sessions
            for k in range(len(cell_list)):
                if cell_list[k] != -1:
                    if not counted: 
                        cnt += 1
                        counted = True
                    cell_x, cell_y = self.footprints[k + 1][cell_list[k]]
                    all_x.extend(cell_x)
                    all_y.extend(cell_y)
                    

            # Convert to numpy arrays
            all_x = np.array(all_x)
            all_y = np.array(all_y)

            # Create a density map
            global image_size
            image_size = 512
            grid_size_x = image_size + 1  
            grid_size_y = image_size + 1  

            density_map = np.zeros((grid_size_y, grid_size_x))
            for x, y in zip(all_x, all_y):
                density_map[y, x] += 1

            # Normalize the density map to create the average footprint
            normalized_footprint = density_map / len(cell_list)

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
            self.global_mask.append(contours)

    def reformat(self):
        new_global_mask = []
        for cell in self.global_mask:
            # Flatten the cell structure
            flattened_cell = [list(point) for point in cell[0]]
            new_global_mask.append(flattened_cell)
        
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


    