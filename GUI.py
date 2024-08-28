import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
from plotting import Plot
import requests
from io import BytesIO
import random 

class QuestionPrompt():
    def __init__(self):
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Question Prompt")
        self.root.geometry("1600x800")  # Set window size

        # Ensure the window is on top and has focus
        self.root.attributes('-topmost', 1)  # Make window always on top
        self.root.lift()  # Bring the window to the top
        self.root.focus_force()  # Force focus to the window

        # Create a label with a question prompt
        self.question_label = tk.Label(self.root, text="Do you want to manually control and correct close cases?", font=("Arial", 24))
        self.question_label.pack(pady=20)  # Add some padding

        # Create a frame to hold the buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=20)

        # Create "Yes" and "No" buttons
        self.yes_button = tk.Button(self.button_frame, text="Yes", font=("Arial", 24), command=self.on_yes)
        self.yes_button.pack(side="left", padx=20)  # Add some padding between buttons

        self.no_button = tk.Button(self.button_frame, text="No", font=("Arial", 24), command=self.on_no)
        self.no_button.pack(side="left", padx=20)
    
    def on_yes(self):
        self.result = "Yes"
        while True:
            threshold = tk.simpledialog.askstring(
                "Input", "What percentage of the best match score should the second-best match be to be considered a close case? (0-100), press cancel to exit", parent=self.root)
            
            if threshold is None:  # Handle cancel operation
                messagebox.showinfo("Operation Cancelled", "Threshold setting has been cancelled.")
                self.result = "No"
                break
            
            if threshold.isdigit():
                messagebox.showinfo("Threshold Set", "{}% will be the threshold for close cases".format(threshold))
                self.threshold = threshold
                break
            else:
                messagebox.showerror("Error", "Please enter a valid number.")
        
        self.root.destroy()  # Ensure window is closed after the loop


    def on_no(self):
        self.result = "No"
        messagebox.showinfo("Manual Correction", "There will be no manual correction!")
        self.root.destroy()  # Close the window after response

    def get_result(self):
        self.root.mainloop()
        return self.result


class ManualCorrection():
    def __init__(self, root, labels, cells_w_aligned_centers, matched_cells, session_cell):
        self.root = root
        self.labels = labels
        self.cells_w_aligned_centers = cells_w_aligned_centers
        self.matched_cells = matched_cells
        self.session_cell = session_cell

        # Ensure the window is on top and has focus
        #self.root.attributes('-topmost', 1)  # Make window always on top
        
       # Create and add the "None" label to the labels list
        self.none_label = tk.Label(self.root, text="None", font=("Arial", 20), bg="white")
        self.none_label.grid(row=0, column=0, padx=20, pady=20)
        self.labels.append(self.none_label)
        
        # Add image to "None" label
        #self.update_none_image()
        
        # Position other labels
        for i, label in enumerate(labels):
            label.grid(row=i // 2, column=i % 2, padx=10, pady=10)
            label.bind("<Button-1>", lambda event, index=i: self.on_click(event, index, cells_w_aligned_centers, matched_cells, session_cell))
       
    def swap(self, lst, index1, index2):
        """Swaps two elements in a list."""
        lst[index1], lst[index2] = lst[index2], lst[index1]

    def on_click(self, event, index, cells_w_aligned_centers, matched_cells, session_cell):
        label = self.labels[index]
        label.config(bg="red")
        gm_cell = label.cget("text")

        if gm_cell.isdigit():
            gm_cell = int(gm_cell)
            message = ("You have selected the best match for the current session "
                    "cell to be the global mask cell: {}").format(gm_cell)
             # Find the index in cells_w_aligned_centers that matches the session_cell
            j = 0
            while cells_w_aligned_centers[j][0][0] != session_cell:
                j += 1

            cells_w_aligned_centers[j] = list(cells_w_aligned_centers[j])
            # Find the index in cells_w_aligned_centers[j] that matches the gm_cell
            k = 1
            while k < len(cells_w_aligned_centers[j]) - 1 and cells_w_aligned_centers[j][k][0] != gm_cell:
                k += 1

            # Perform the swap
            self.swap(cells_w_aligned_centers[j], k, 0)  # Swap with the first element in cells_w_aligned_centers[j]
            
            # Get the updated coordinate
            coord = cells_w_aligned_centers[j][0][1]
            coord = list(coord)

            # Update the matched_cells with the new coordinate
            i = 0
            while matched_cells[i][0][0] != session_cell:
                i += 1
            
            # Ensure matched_cells[i][1] is mutable
            matched_cells[i] = list(matched_cells[i])
            matched_cells[i][1] = list(matched_cells[i][1]) if isinstance(matched_cells[i][1], tuple) else matched_cells[i][1]
            
            matched_cells[i][1][1] = coord
        else:
            message = ("You have chosen none of the cells to be a good match for the current session cell.")
            # Ensure matched_cells[i][1] is mutable
            matched_cells[session_cell] = list(matched_cells[session_cell])
            matched_cells[session_cell][1] = list(matched_cells[session_cell][1]) if isinstance(matched_cells[session_cell][1], tuple) else matched_cells[session_cell][1]
            matched_cells[session_cell][1][0], matched_cells[session_cell][1][1] = None, None

        # Display the message box
        messagebox.showinfo("Selection", message)        
        label.unbind("<Button-1>")
        label.bind("<Button-1>", lambda event, index=index: self.on_unclick(event, index))
  

    def update_none_image(self, size=(200, 200)):
        # Fetch a random image from Pexels
        query = 'magic brain psychedelics'  # or 'AI', 'technology', etc.
        url = f"https://www.pexels.com/search/{query}/?page={random.randint(1, 10)}"  # Random page        
        
        
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        
        # Extract image URL from the HTML
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        image_tags = soup.find_all('img', {'class': 'photo-item__img'})
        
        if image_tags:
            image_url = random.choice(image_tags)['src']
            
            # Download the image
            response = requests.get(image_url)
            img_data = BytesIO(response.content)
            
            # Open image with PIL
            img = Image.open(img_data)
            img = img.resize((200, 200))  # Resize to fit label
            img = ImageTk.PhotoImage(img)
            
            # Update the label with the image
            self.none_label.config(image=img)
            self.none_label.image = img  # Keep a reference to avoid garbage collection
        else:
            print("No images found or error in fetching images.")