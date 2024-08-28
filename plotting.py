import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage import measure
import os
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from skimage.measure import find_contours
from skimage.draw import polygon


class Plot():
    def __init__(self, mask, binary_mask=None, output_dir=None):
        self.mask = mask
        self.binary_mask = binary_mask
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_mask(self, title, idx=None, output_dir=None, session_cell=None, session_pixels=None, overlap_score = None, session=None, alignment_score=None):
        plt.figure()
        plt.axis('equal')

        # adapt output directory
        if session_cell is not None and session_pixels is not None and session is not None:
            output_dir = os.path.join(self.output_dir, 'Session ' + str(session))
            os.makedirs(output_dir, exist_ok=True)

        # Flatten the mask if it's a list of lists or nested structure
        if isinstance(self.mask, np.ndarray):
            self.mask = self.mask.tolist()

        x_coords, y_coords, x_coords_cell, y_coords_cell = [], [], [], []    

        # Iterate through each cell in the mask and extract the pixel coordinates
        for cnt, cell in enumerate(self.mask):
            if isinstance(cell, (np.ndarray, list)) and cell.ndim == 2 and cell.shape[1] == 2:
                for pixel in cell:
                    if len(pixel) == 2:
                        x_coords.append(pixel[0])
                        y_coords.append(pixel[1])
                        if idx and cnt == idx:
                            x_coords_cell.append(pixel[0])
                            y_coords_cell.append(pixel[1])
                        if idx and cnt > idx:
                            break
        
        if session_cell is not None and session_pixels is not None and idx is not None:
            file_path = self.plot_session_cell_and_corresponding_gm_cells(idx, title, session_cell, x_coords_cell, y_coords_cell, session_pixels, output_dir)
            if overlap_score is not None:
                file_path = self.plot_session_cell_and_corresponding_gm_cells(idx, title, session_cell, x_coords_cell, y_coords_cell, session_pixels, output_dir, overlap_score=overlap_score)
            if alignment_score is not None:
                file_path = self.plot_session_cell_and_corresponding_gm_cells(idx, title, session_cell, x_coords_cell, y_coords_cell, session_pixels, output_dir, alignment_score=alignment_score)
            if alignment_score is not None and overlap_score is not None:
                file_path = self.plot_session_cell_and_corresponding_gm_cells(idx, title, session_cell, x_coords_cell, y_coords_cell, session_pixels, output_dir, overlap_score=overlap_score, alignment_score=alignment_score)

        else:
            plt.scatter(y_coords, x_coords, c='b', s=1)
            plt.title(title)

            output_dir = os.path.join(output_dir, 'General')
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f'{title}.png')

        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        self.cell_pair_path = file_path
        #plt.show()
        plt.close()

    def plot_session_cell_and_corresponding_gm_cells(self, idx, title, session_cell, gm_cells_x_coord, gm_cells_y_coord, session_pixels, output_dir, overlap_score=None, alignment_score=None):
        y_coords_session_cell = session_pixels[session_cell][0]
        x_coords_session_cell = session_pixels[session_cell][1]

        # Create a binary mask for the session cell
        max_y = max(y_coords_session_cell) + 2
        max_x = max(x_coords_session_cell) + 2
        binary_image = np.zeros((max_y, max_x), dtype=bool)
        rr, cc = polygon(y_coords_session_cell, x_coords_session_cell)
        binary_image[rr, cc] = True

        # Extract contours
        contours = find_contours(binary_image, level=0.5)

        # Plot contours
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], 'c', linewidth=2)  # Line plot for contours


        # Plot pixel contours of global mask with interpolation
        if len(gm_cells_x_coord) > 1 and len(gm_cells_y_coord) > 1 and len(x_coords_session_cell) > 1 and len(y_coords_session_cell) > 1:
            t = np.arange(len(gm_cells_x_coord))
            t_new = np.linspace(t.min(), t.max(), 300)
            
            interp_func_x = interp1d(t, gm_cells_x_coord, kind='linear')
            interp_func_y = interp1d(t, gm_cells_y_coord, kind='linear')

            x_smooth = interp_func_x(t_new)
            y_smooth = interp_func_y(t_new)
            
            plt.plot(y_smooth, x_smooth, 'm')  
        else:
            plt.plot(gm_cells_y_coord, gm_cells_x_coord, 'm')  

        plt.title(f'{title} Cell {idx} vs Session Cell {session_cell}')
        
        if overlap_score is not None:
            plt.title(f'{title} Cell {idx} vs session cell {session_cell} with overlap: {overlap_score:.2f}')
        if alignment_score is not None:
            plt.title(f'{title} Cell {idx} vs session cell {session_cell} with match score: {alignment_score:.2f}')
        if overlap_score is not None and alignment_score is not None:
            plt.title(f'{title} Cell {idx} vs session cell {session_cell} with match score: {alignment_score:.2f}')
        
        # Ensure output directory exists
        output_dir = os.path.join(output_dir, 'Session_Cell_' + str(session_cell))
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'{title}_cell_{idx}.png')
        return file_path
    
    def extract_contours(self, image):
        # Threshold the image if needed to get binary mask
        binary_image = image > 0
        # Find contours at a constant value of 0.5
        contours = measure.find_contours(binary_image, level=0.5)
        return contours

    def plot_contours(self, image, contours, name, cnt, output_dir, gamma=3.0, dpi=300):
        plt.figure(figsize=(8, 8))
        
        # Normalize the image for better contrast
        image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        image_corrected = np.power(image_normalized, gamma)
        plt.imshow(image_corrected, cmap='gray', vmin=0, vmax=1)
        
        # Plot each contour
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
        
        # Set the title and hide axes
        title = f'Contours_{name}_{cnt}'
        plt.title(title)
        plt.axis('off')
        
        # Define the file path for saving the plot
        output_file_path = os.path.join(output_dir, f"{title}.png")
        
        # Save the plot
        plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.show()
        plt.close()
        print(f"Plot saved to {output_file_path}")

    def plot_tiff(self, file_path):
        # Read the TIFF file
        with tifffile.TiffFile(file_path) as tif:
            # For multi-page TIFF, read the first page
            image = tif.pages[0].asarray()
            
            # Display the image
            plt.figure(figsize=(8, 8))
            plt.imshow(image, cmap='inferno')
            plt.title(f'Image from {file_path}')
            plt.axis('off')  # Turn off axis numbers and ticks
            plt.show()
            plt.close()

    def plot_binary_mask(self, title, output_dir):
        # Plot the binary mask
        plt.imshow(self.binary_mask, cmap='gray')
        plt.title(title)
        plt.axis('off') 
        output_file_path = os.path.join(output_dir, f"{title}.png")
        plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close()

    def plot_filled_cells(self, mask, centroids=None):
        plt.figure(figsize=(8, 8))
        plt.imshow(mask, cmap='gray')

        if centroids:
            for centroid in centroids:
                if len(centroid) == 2:
                    plt.plot(centroid[1], centroid[0], 'ro', markersize=2)  # Red dot for the centroid
                    #plt.text(centroid[0], centroid[1], f'C{i}', color='red', fontsize=6, ha='center')
                    plt.title('Filled Cells with centroids')
        else:
            plt.title('Filled Cells')
        plt.axis('off')
        #plt.show()

    def plot_cells_with_overlap(self, overlap_mapping, original_cells, filled_global_cells):
        plt.figure(figsize=(10, 10))

        # Define colors for original cells, global cells, and overlaps
        original_color = 'blue'
        global_color = 'green'
        overlap_color = 'red'
        
        # Plot original cells
        for cell_key, (ypix, xpix) in original_cells.items():
            plt.scatter(xpix, ypix, color=original_color, s=1, label=f"{cell_key} (Original)")
        
        # Plot global cells
        for cell_key, (ypix, xpix) in enumerate(filled_global_cells):
            plt.scatter(xpix, ypix, color=global_color, s=1, label=f"{cell_key} (Global)")

        # Plot overlaps
        for original_key, overlaps in overlap_mapping.items():
            original_ypix, original_xpix = original_cells[original_key]

            for overlap_ratio, global_key in overlaps:
                global_ypix, global_xpix = filled_global_cells[global_key]

                original_set = set(zip(original_ypix, original_xpix))
                global_set = set(zip(global_ypix, global_xpix))

                overlap_set = original_set.intersection(global_set)

                if overlap_set:
                    overlap_ypix, overlap_xpix = zip(*overlap_set)
                    plt.scatter(overlap_xpix, overlap_ypix, color=overlap_color, s=1, label=f"Overlap: {original_key} & {global_key}")

        # Create proxy artists for the legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=original_color, markersize=10, label='Original Cells'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=global_color, markersize=10, label='Global Cells'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=overlap_color, markersize=10, label='Overlap')
        ]

        plt.legend(handles=legend_elements, loc='upper right')
        plt.title('Cells with Overlaps')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.savefig(os.path.join(self.output_dir, 'cells_with_overlaps.png'), bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close()

    def plot_overlap_distribution(self, session_cell, overlap_cells, output_dir):
        reset = os.getcwd()
        # Plot the overlap distribution for each overlapping cell
        plt.figure(figsize=(12, 6))
        overlap_values = [overlap[0] for overlap in overlap_cells]
        indices = [overlap[1] for overlap in overlap_cells]

        bar_width = 0.35
        x_positions = range(len(indices))
        bars = plt.bar(x_positions, overlap_values, color='blue', alpha=0.7, width=bar_width)

        # Annotate each bar with its height (overlap value)
        for bar, idx in zip(bars, indices):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom')

        plt.xticks(x_positions, indices)
        plt.xlim(-0.5, len(overlap_cells) - 0.5)  # Adjust x-axis limits to fit bars within fixed plot size
        plt.xlabel('Cell Index')
        plt.ylabel('Overlap Value')
        plt.title('Bar Plot of Overlap Values for Cell ' + str(session_cell))
        directory_path = os.path.abspath(os.path.join(output_dir, 'overlap_distributions'))

        # Create the directory and change to it
        os.makedirs(directory_path, exist_ok=True)  # os.makedirs can create intermediate directories if they don't exist
        os.chdir(directory_path)
        plt.savefig(os.path.join(os.getcwd(), f'Overlap_Distribution_Cell_{session_cell}.png'))
        #plt.show()
        plt.close()
        os.chdir(reset)

    def plot_cells_w_aligned_centers(self, cells_aligned_centers, title, session_cell=None, session=None):
        plt.figure(figsize=(10, 10))
        colors = ['r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        legend_entries = set()
        if session_cell is not None and session is not None: 
            #output_dir = self.output_dir
            output_dir = os.path.join(self.output_dir, 'Session ' + str(session))
            os.makedirs(output_dir, exist_ok=True)
            # Ensure output directory exists
            output_dir = os.path.join(output_dir, 'Session_Cell_' + str(session_cell))
            os.makedirs(output_dir, exist_ok=True)
        else: 
            output_dir = self.output_dir

        for (i, centered_session_cell_contour), (j, global_mask_cell_contour) in cells_aligned_centers:
            
            if session_cell is not None and i != session_cell:
                continue
            
            if session_cell is not None and i == session_cell:
                color = colors[j % len(colors)]  # Cycle through the colors
            else:
                color = 'm'

            y_coords_session_cell = [y for x, y in centered_session_cell_contour]
            x_coords_session_cell = [x for x, y in centered_session_cell_contour]

            max_y = max(y_coords_session_cell) + 2
            max_x = max(x_coords_session_cell) + 2
            binary_image = np.zeros((max_x, max_y), dtype=bool)
            rr, cc = polygon(x_coords_session_cell, y_coords_session_cell)
            binary_image[rr, cc] = True
            session_contours = find_contours(binary_image, level=0.5)

            for contour in session_contours:
                if f"Session Cell {i}" not in legend_entries:
                    plt.plot(contour[:, 1], contour[:, 0], 'c', linewidth=2, label=f"Session Cell {i}")
                    legend_entries.add(f"Session Cell {i}")
                else:
                    plt.plot(contour[:, 1], contour[:, 0], 'c', linewidth=2)
            
            
            y_coords_global_mask = [y for y, x in global_mask_cell_contour]
            x_coords_global_mask = [x for y, x in global_mask_cell_contour]

            if f"Global Mask Cell {j}" not in legend_entries:
                plt.plot(x_coords_global_mask, y_coords_global_mask, color=color, linewidth=2, label=f"Global Mask Cell {j}")
                legend_entries.add(f"Global Mask Cell {j}")
            else:
                plt.plot(x_coords_global_mask, y_coords_global_mask, color=color, linewidth=2)

        plt.title(title)
        if session_cell is not None:
            plt.legend(loc='upper right')
        plt.axis('equal')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"{title}.png"), bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close()

    def plot_contours(self, session_cell, overlap_cell):
        # Plot the binary mask
        print(self.binary_mask.shape)
        plt.imshow(self.binary_mask[overlap_cell[1]], cmap='gray', rotation=90)
        plt.title("test")
        plt.axis('off') 
        #output_file_path = os.path.join(output_dir, f"{title}.png")
        #plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close()
    
    def plot_dtw_alignment(self, session_cell_coord, global_cell_coord, dtw_path, session_cell_idx, global_cell_idx):
        plt.figure(figsize=(10, 10))
        plt.axis('equal')
             
        session_cell_coord = np.array(session_cell_coord)
        global_cell_coord = np.array(global_cell_coord)
        output_dir = os.path.join(self.output_dir, 'DTW_Alignment')
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, 'Session_Cell_' + str(session_cell_idx))
        os.makedirs(output_dir, exist_ok=True)
        

        for (i, j) in dtw_path:
            plt.plot([session_cell_coord[i][1], global_cell_coord[j][1]], 
                    [session_cell_coord[i][0], global_cell_coord[j][0]], 'r-')
        
         # Plot the session cell coordinates with rotation
        plt.plot(
            session_cell_coord[:, 1],  # Swap x and y
            session_cell_coord[:, 0],   
            'bo-', 
            label='Session Cell {}'.format(session_cell_idx)
        )
        
        # Plot the global cell coordinates with rotation
        plt.plot(
            global_cell_coord[:, 1],    # Swap x and y
            global_cell_coord[:, 0],    
            'go-', 
            label='Global Cell {}'.format(global_cell_idx)
        )
        plt.legend()
        title = 'DTW Alignment Session Cell {} with GM cell {}'.format(session_cell_idx, global_cell_idx)
        plt.title('DTW Alignment')
        plt.savefig(os.path.join(output_dir, f"{title}.png"), bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close()
        
    def plot_distribution(self, array, title=None):
        plt.figure()
        plt.boxplot(array)
        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel('Shape Similarity')
        plt.savefig(os.path.join(self.output_dir, f"{title}.png"), bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close()

    def plot_distribution_of_overlap_number(self, same_cells, session):
        overlapping_cells = [len(cells) for cells in same_cells.values()]

        bin_edges = range(1, max(overlapping_cells) + 2)
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

        plt.figure()
        plt.hist(overlapping_cells, bins=bin_edges, color='blue', alpha=0.7)
        plt.title('Distribution of Overlapping Cells for session ' + str(session))
        plt.xlabel('Number of Overlapping Cells')
        plt.ylabel('Frequency')

        plt.xticks(ticks=bin_centers, labels=range(1, len(bin_centers) + 1))
        plt.savefig(os.path.join(self.output_dir, 'overlap_distribution.png'), bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close()

    def plot_downsampled_sequences(self, seq1, seq2):
        plt.figure()
        plt.plot(seq1[:, 0], seq1[:, 1], 'bo-', label='Downsampled Seq 1')
        plt.plot(seq2[:, 0], seq2[:, 1], 'go-', label='Downsampled Seq 2')
        plt.legend()
        plt.show()