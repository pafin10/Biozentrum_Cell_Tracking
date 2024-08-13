import numpy as np
import matplotlib.pyplot as plt
import h5py
from collections import defaultdict

class CellRegGlobalMask:
    def __init__(self, matfile):
        self.load_mat(matfile)
        self.footprints = self.convert_to_footprints(self.binary_footprints)
        self.average_footprints()
    
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
        for cell, cell_list in self.mapping.items():
            cnt = 0
            if cell != -1:
                x_list, y_list = [self.footprints[0][cnt][0]], [self.footprints[0][cnt][1]]
                cnt += 1
                for k in range(1, len(cell_list)):
                    if cell_list[k] != -1:
                        cell_contours = self.footprints[k][cell_list[k]]
                        x_list.append(cell_contours[0])
                        y_list.append(cell_contours[1])
                self.avg_footprints = [[np.mean(x_list[i]) for i in range(len(x_list))], [np.mean(y_list[i]) for i in range(len(y_list))]]

if __name__ == '__main__':
    matfile = r'H:\Desktop\Code\Scores and Alignment\cell_reg\cellRegistered_20240812_165323.mat'
    cr = CellRegGlobalMask(matfile)
    print(cr.centroids.shape) # (2 x n_cells)
    for k,v in cr.mapping.items():
        print(k, v) # (n_sessions x n_cells)
    print(cr.footprints[0][0]) # (n_cells x 512 x 512)
    print(cr.avg_footprints)

        
    # Display the first cell footprint
    #plt.imshow(cr.footprints[0][1])
    #plt.show()


    