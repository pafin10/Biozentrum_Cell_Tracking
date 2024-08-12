import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
import os
from collections import defaultdict
from skimage.measure import find_contours, centroid
from skimage.draw import polygon


def dist(centroid, centroid1):
    return ((centroid[0] - centroid1[0]) ** 2 + (centroid[1] - centroid1[1]) ** 2) ** 0.5

def load_files(root_dir):
    statsfiles = []
    opsfiles = []
    cellfiles = []

    # Load data
    print("Loading data...")

    # Iterate over the directories in the root directory
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            if dir.startswith('plane0'):
                # Construct the full directory path
                full_dir_path = os.path.join(root, dir)

                # Load ops.npy file
                ops_path = os.path.join(full_dir_path, 'ops.npy')
                if os.path.exists(ops_path):
                    ops = np.load(ops_path, allow_pickle=True).item()
                    opsfiles.append(ops)

                # Load stat.npy file
                stat_path = os.path.join(full_dir_path, 'stat.npy')
                if os.path.exists(stat_path):
                    stat = np.load(stat_path, allow_pickle=True)
                    statsfiles.append(stat)

                cell_path = os.path.join(full_dir_path, 'iscell.npy')
                if os.path.exists(cell_path):
                    cell = np.load(cell_path, allow_pickle=True)
                    cellfiles.append(cell)

    return statsfiles, opsfiles, cellfiles


def convert_to_binary_mask(mask, image_shape):
    binary_mask = np.zeros(image_shape, dtype=np.uint8)
    for cell in mask:
        for pixel in cell:
            if isinstance(pixel, (list, np.ndarray)) and len(pixel) == 2:
                x, y = int(pixel[0]), int(pixel[1])
                if 0 <= x < image_shape[0] and 0 <= y < image_shape[1]:
                    binary_mask[x, y] = 1
    return binary_mask

def raw_to_tiff(fname):
    ##########################################################
    ########## CONVERT BSCOPE RAW DATA TO TIFF ###############
    ##########################################################
    data = np.memmap(fname,
                dtype='uint16',
                mode='r')
    data = data.reshape(-1, 512,512)
    fname_out = fname.replace('.raw','.tiff')
    tifffile.imwrite(fname_out, data)
    #
    print ("...DONE...")

def save_plot(plot, plot_name):
    plt.imshow(plot)
    plt.title(plot_name)
    plt.savefig(f'{plot_name}.png') 
    plt.close()  

def save_overlap_plot(im_overlap, cell1_index, cell2_index, filename):
    plt.imshow(im_overlap, cmap='hot')  
    plt.title(f'Overlap between cell {cell1_index} and cell {cell2_index}')
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

def shift_pixels(pixels, shift):
    shifted_pixels = [round(p + shift) for p in pixels]
    return shifted_pixels

def calculate_overlap(cell1_ypix, cell1_xpix, cell2_ypix, cell2_xpix):
    # Sort pixel coordinates to avoid mismatch
    cell1_pixels = sorted(zip(cell1_ypix, cell1_xpix))
    cell2_pixels = sorted(zip(cell2_ypix, cell2_xpix))
    
    cell1_set = set(cell1_pixels)
    cell2_set = set(cell2_pixels)
    
    overlap_pixels = cell1_set.intersection(cell2_set)
    union_pixels = cell1_set.union(cell2_set)   
    overlap_ratio = len(overlap_pixels) / len(union_pixels)

    return overlap_ratio

def fill_cells_gm(contours, image_shape):
    filled_areas = []
    mask = np.zeros(image_shape, dtype=np.uint8)

    for contour in contours:
        contour = [(y, x) for x, y in contour]
        contour = np.array(contour).reshape((-1, 1, 2)).astype(np.int32)

        # Draw and fill the contour on an individual mask
        individual_mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(individual_mask, [contour], -1, (255), thickness=cv2.FILLED)

        # Find the filled area for the current contour
        filled_pixels = np.transpose(np.nonzero(individual_mask))

        filled_areas.append(filled_pixels.tolist())
        # Add individual mask to the main mask
        mask = np.maximum(mask, individual_mask)

    return mask, filled_areas

def compute_centroids(filled_cells):
    centroids = defaultdict(list)

    for i, area in enumerate(filled_cells):
        area = np.array(area)
        if area.shape[0] == 0:
            continue  # Skip empty areas
            
        # Create a binary image from the cell area
        max_y, max_x = area.max(axis=0) + 1
        binary_image = np.zeros((max_y, max_x), dtype=np.uint8)
        binary_image[area[:, 0], area[:, 1]] = 1
        
        # Compute centroid of the binary image
        centroid_y, centroid_x = centroid(binary_image)
        
        # Store the centroid
        centroids[i].append([centroid_y, centroid_x])

    return centroids

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))
           
def convert_global_mask_to_list_structure(global_mask):
    global_mask_list = []
    for cell in global_mask:
        coords = []
        if isinstance(cell, (np.ndarray, list)) and cell.ndim == 2 and cell.shape[1] == 2:
            for pixel in cell:
                if len(pixel) == 2:
                    coords.append([int(pixel[0]), int(pixel[1])])
        global_mask_list.append(coords)
    return global_mask_list

def compute_cell_alignment_score(overlap, dtw_distance, centered_overlap_score, weights=[0.2, 0.3, 0.5], using_dtw=True, n_cells=None):
    if using_dtw:
        dtw_distance = 1 - dtw_distance # we want higher scores to be better
    
    if n_cells is not None:
        if n_cells == 2 and not using_dtw:
            return overlap * 0.3 + centered_overlap_score * 0.7
        elif n_cells == 3:
            if not using_dtw:
                return overlap * 0.2 + centered_overlap_score * 0.8
            else: 
                weights = [0.2, 0.4, 0.4]
        else:
            weights = [0.2, 0.3, 0.5]
    
    if using_dtw:
        return overlap * weights[0] + dtw_distance * weights[1] + centered_overlap_score * weights[2]

def extract_contours(session_cell_pixels):
    session_cells_contours = []
    for cell_pixels in session_cell_pixels: 
        y_coords = [y for x, y in cell_pixels]
        x_coords = [x for x, y in cell_pixels]

        max_y = max(y_coords) + 2
        max_x = max(x_coords) + 2
        binary_image = np.zeros((max_y, max_x), dtype=bool)
        rr, cc = polygon(y_coords, x_coords)
        binary_image[rr, cc] = True
        session_contours = find_contours(binary_image, level=0.5)
        contours = [contour.astype(np.int32) for contour in session_contours]
        session_cells_contours.append(contours)

    unpacked_contours = []
    for contour in session_cells_contours:
        contour = contour[0]
        unpacked_contour = [[int(coord[1]), int(coord[0])] for coord in contour]
        # avoid duplicate pixels for potentially more accurate calculation of warping path
        unpacked_contour = list(dict.fromkeys(map(tuple, unpacked_contour)))
        unpacked_contours.append(unpacked_contour)
    return unpacked_contours

def fill_cells(contours, image_shape):


    """
    Given cell contours, return arrays of x pixels and y pixels that fill the area.

    Parameters:
    - contours: A list of contours, where each contour is itself a list of x and y pixel coordinates.
                Example: [[x1, x2, ...], [y1, y2, ...]]
    - image_shape: The shape of the image (height, width) within which the contours are defined.

    Returns:
    - x_pixels: A numpy array of x pixel coordinates that fill the contour area.
    - y_pixels: A numpy array of y pixel coordinates that fill the contour area.
    """

    # Create an empty binary image (mask) with the given image shape
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert contours into the format expected by cv2.fillPoly
    # cv2.fillPoly expects a list of polygons, where each polygon is a Nx2 array of (x, y) points
    contours = np.array(contours).T.reshape(-1, 1, 2)  # Reshape to (N, 1, 2) format

    # Fill the contour area on the mask
    cv2.fillPoly(mask, [contours], 1)

    # Extract the x and y pixel coordinates where the mask is filled
    y_pixels, x_pixels = np.where(mask == 1)

    return x_pixels, y_pixels

def generate_cyclic_permutations(arr):
    permutations = []
    for i in range(len(arr)):
        perm = arr[i:] + arr[:i]
        permutations.append(perm)
    
    return permutations