import tkinter as tk
from tkinter import messagebox, simpledialog

class QuestionPrompt():
    def __init__(self):
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Question Prompt")
        self.root.geometry("800x400")  # Set window size

        # Create a label with a question prompt
        self.question_label = tk.Label(self.root, text="Do you want to manually control and correct close cases?", font=("Arial", 16))
        self.question_label.pack(pady=20)  # Add some padding

        # Create a frame to hold the buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)

        # Create "Yes" and "No" buttons
        self.yes_button = tk.Button(self.button_frame, text="Yes", font=("Arial", 14), command=self.on_yes)
        self.yes_button.pack(side="left", padx=10)  # Add some padding between buttons

        self.no_button = tk.Button(self.button_frame, text="No", font=("Arial", 14), command=self.on_no)
        self.no_button.pack(side="left", padx=10)
    
    def on_yes(self):
        self.result = "Yes"
        threshold = tk.simpledialog.askstring(
            "Input", "What percentage of the best match score should the second-best match be to be considered a close case? (0-100)")
        if threshold is not None:  # Check if the user provided a value
            messagebox.showinfo("Threshold Set", "{} % will be the threshold for close cases".format(threshold))
            self.threshold = threshold
        self.root.destroy()  # Close the window after response

    def on_no(self):
        self.result = "No"
        messagebox.showinfo("Manual Correction", "There will be no manual correction!")
        self.root.destroy()  # Close the window after response

    def get_result(self):
        self.root.mainloop()
        return self.result


class ManualCorrection():
    def __init__(self):
        # Create the main window
        self.root = tk.Tk()
        
        return self.result