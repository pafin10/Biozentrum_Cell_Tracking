import hausdorff
import numpy as np
import os
from utils import *
from tslearn.metrics import dtw, dtw_path, soft_dtw, dtw_path_from_metric
from typing import List


class Shapes():
    def __init__(self, global_mask_cell_contours, session_cell_contours):
        self.global_mask_cell_contours = global_mask_cell_contours
        self.session_cell_contours = session_cell_contours
        self.dtw = None
    
    def translate(self, contour, xdiff, ydiff):
        y_pix = [int(round(y - ydiff)) for [x,y] in contour]
        x_pix = [int(round(x - xdiff)) for [x,y] in contour]
        return [[x,y] for x,y in zip(x_pix, y_pix)]
    
    def align_centers(self, gm_centroids, session_centroids):
        cells_aligned_centers = []
        for i, session_centroid in session_centroids.items():
            session_centroid = session_centroid[0] 
            for j, gm_centroid in enumerate(gm_centroids): 

                if dist(session_centroid, gm_centroid) < 20:
                    xdiff = session_centroid[0] - gm_centroid[0]
                    ydiff = session_centroid[1] - gm_centroid[1]
                    session_cell_contour = self.session_cell_contours[i]
                    centered_session_cell_contour = self.translate(session_cell_contour, xdiff, ydiff)
                    cells_aligned_centers.append(((i, centered_session_cell_contour), (j, self.global_mask_cell_contours[j])))
        
        return cells_aligned_centers
    
    class DTW():
        def __init__(self, cell1, cell2, threshold=5):
            self.cell1 = cell1
            self.cell2 = cell2
            self.dtw_path = None
            self.dtw_dist = None
            self.soft_dtw_dist = None
            self.threshold = threshold
            self.computeDTW()
        
        def computeDTW(self):
            
            self.dtw_path, dtw_dist = dtw_path(self.cell1, self.cell2, global_constraint='itakura', itakura_max_slope=10)
            self.dtw_dist = dtw_dist / len(self.dtw_path)
            self.soft_dtw_dist = soft_dtw(self.cell1, self.cell2, gamma=0.1)

        # did not help to find extreme shapes and takes 4 times longer
        def quadratic_penalty_distance(self, A, B):
            distance = euclidean_distance(A, B)
            if distance > self.threshold:
                # Apply a quadratic penalty for distances greater than the threshold
                return (distance ** 2)
            else:
                # Normal distance for distances within the threshold
                return distance


    