import numpy as np
import scipy.io

def extract_roi_footprints(statsfile, ops_file, cellfile, output_mat_file):
    # Load the stat.npy file
    stat = np.load(statsfile, allow_pickle=True)
    ops = np.load(ops_file, allow_pickle=True).item()
    cellfile = np.load(cellfile, allow_pickle=True)

    # Extract ROI footprints
    n = len(stat)
    session_cells = [(stat[i], i) for i in range(n) if cellfile[i][0]]

    # Extract the x and y coordinates of the pixels for each selected cell
    filled_cells = [(session_cells[i][0]['xpix'], session_cells[i][0]['ypix']) for i in range(len(session_cells))]

    # Save the footprints to a .mat file
    if filled_cells:
        print(filled_cells[0])
        scipy.io.savemat(output_mat_file, {'filled_cells': filled_cells})
    else:
        print("No cells found in the session")

if __name__ == '__main__':
    filled_cells = extract_roi_footprints(
        statsfile=r'H:\Desktop\Code\DON-019539_B\20240522\plane0\stat.npy',
        ops_file=r'H:\Desktop\Code\DON-019539_B\20240522\plane0\ops.npy', 
        cellfile=r'H:\Desktop\Code\DON-019539_B\20240522\plane0\iscell.npy',
        output_mat_file='footprints.mat'
    )
