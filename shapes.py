import hausdorff
import numpy as np
import os
from utils import *
from frechetdist import frdist
from dtw import dtw
from typing import List


class Shapes():
    def __init__(self, global_mask_cell_contours, session_cell_contours):
        self.global_mask_cell_contours = global_mask_cell_contours
        self.session_cell_contours = session_cell_contours
    
    def translate(self, contour, xdiff, ydiff):
        y_pix = [int(round(y - ydiff)) for [x,y] in contour]
        x_pix = [int(round(x - xdiff)) for [x,y] in contour]
        return [[x,y] for x,y in zip(x_pix, y_pix)]
    
    def align_centers(self, gm_centroids, session_centroids):
        cells_aligned_centers = []
        for i, session_centroid in enumerate(session_centroids): 
            for j, gm_centroid in gm_centroids.items(): 
                gm_centroid = gm_centroid[0]

                if dist(session_centroid, gm_centroid) < 20:
                    xdiff = session_centroid[0] - gm_centroid[0]
                    ydiff = session_centroid[1] - gm_centroid[1]
                    session_cell_contour = self.session_cell_contours[i]
                    centered_session_cell_contour = self.translate(session_cell_contour, xdiff, ydiff)
                    cells_aligned_centers.append(((i, centered_session_cell_contour), (j, self.global_mask_cell_contours[j])))
        
        return cells_aligned_centers
    
    def calculate_shape_similarity(self, cell1 : List, cell2 : List):
        return dtw(cell1, cell2).normalizedDistance