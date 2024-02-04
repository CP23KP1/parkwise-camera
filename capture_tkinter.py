import tkinter as tk
from tkinter import ttk

def on_select():
    selected_value = combo.get()
    print("Selected:", selected_value)

# Create the main window
window = tk.Tk()

# Set the window title
window.title("Dropdown Window")

# Set the window size (optional)
window.geometry("400x300")

# Set the background color to white
window.configure(bg="white")

data_list = ["Option 1", "Option 2", "Option 3", "Option 4"]

# Create a StringVar to store the selected value
selected_value = tk.StringVar()

# Create a Combobox and set its values
combo = ttk.Combobox(window, values=data_list, textvariable=selected_value)
combo.place(relx=0.5, rely=0.4, anchor="center")  # Place the combobox in the center

# Set a default value for the combobox (optional)
combo.set("Select an option")

# Create a button to trigger an action based on the selected item
select_button = tk.Button(window, text="Select", command=on_select)
select_button.place(relx=0.5, rely=0.6, anchor="center")  # Place the button below the combobox

window.mainloop()
