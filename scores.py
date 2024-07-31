import pickle
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage import measure
from tqdm import tqdm
from utils import *
from plotting import Plot
import os


def get_nested_list_shape(nested_list):
    dimensions = []
    current_level = nested_list

    while isinstance(current_level, list) or isinstance(current_level, np.ndarray):
        dimensions.append(len(current_level))
        current_level = current_level[0]
        
        # If the current level is a NumPy array with dtype=object, keep iterating
        if isinstance(current_level, np.ndarray) and current_level.dtype == object:
            current_level = current_level.tolist()

    return tuple(dimensions)


if __name__ == '__main__':
    # File paths
    global_mask_file_path = r'H:\Desktop\Code\DON-019539_B\master_mask_GUI.pkl'
    session_mask_file_path = r'H:\Desktop\Code\DON-019539_B\session_mask_GUI.pkl'
    alignment_npz = r'H:\Desktop\Code\DON-019539_B\20240523\alignment\alignment_parameters.npz'
    tiff22 = r'D:\DON-019539_20240522_002P-F_1_no_VR_S1.tiff'
    tiff23 = r'D:\DON-019539_20240523_002P-F_1_no_VR_S1.tiff'
    output_dir = r'D:\Outputs\Alignment_DON-019539'

    # Load masks
    with open(global_mask_file_path, 'rb') as file:
        global_mask = pickle.load(file)
    with open(session_mask_file_path, 'rb') as file:
        session_mask = pickle.load(file)

    # Load the alignment parameters
    alignment_parameters = np.load(alignment_npz)
    for file in alignment_parameters.files:
        print(file, alignment_parameters[file])

    # Convert the global mask to a NumPy array with dtype=object
    np_global_mask = np.array(global_mask, dtype=object)
    np_session_mask = np.array(session_mask, dtype=object)

    # Get the shape of the nested list
    shape_glob = get_nested_list_shape(np_global_mask)
    shape_sess = get_nested_list_shape(np_session_mask)

    print("Shape of the Global Mask:", shape_glob)
    print("Shape of the Session Mask:", shape_sess)
    #plot_mask(np_global_mask)
    #plot_mask(np_session_mask)
    #raw_to_tiff(r'H:\Desktop\Code\DON-019539_B\20240523\plane0\DON-019539_20240523_002P-F_1_no_VR_S1.raw')
    #raw_to_tiff(r'H:\Desktop\Code\DON-019539_B\20240522\plane0\DON-019539_20240522_002P-F_1_no_VR_S1.raw')

    # Load images
    tiff_files = [tiff22, tiff23]

    # Save tiff files as npy
    """"
    for tiff_file in tqdm(tiff_files, desc="Loading images"):
        image = tifffile.imread(tiff_file)
        images.append(image)
        print(f"Loaded {tiff_file} with shape {image.shape}")

    # Save images
    for i, image in enumerate(images):
        np.save(f'image_{i}.npy', image)
    """
    
    # Extract contours and plot first n frames of original recordings  
    n = 10
    recordings = [r'D:\image_0.npy', r'D:\image_1.npy']
    # improve plotting ie clearer contrast, sharper picture
    for recording in recordings:
        name = os.path.basename(recording).split('_')[1]  # Extract name from file path        
        recording = np.load(recording)
        cnt = 0
        for image in recording:
            if cnt == n:
                break
            P = Plot(image)
            contours = P.extract_contours(image)
            P.plot_contours(image, contours, name, cnt, output_dir=output_dir)
            cnt += 1
    

    # Convert masks to binary
    shape = (512, 512)
    binary_global_mask = convert_to_binary_mask(np_global_mask, shape)
    binary_session_mask = convert_to_binary_mask(np_session_mask, shape)

    # Plot binary masks
    P_gl = Plot(binary_global_mask)
    P_se = Plot(binary_session_mask)
    P_gl.plot_binary_mask(binary_global_mask,'Global Mask', output_dir=output_dir)
    P_se.plot_binary_mask(binary_session_mask, 'Session Mask', output_dir=output_dir)


