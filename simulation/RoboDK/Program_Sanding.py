import os
from robodk import *      # RoboDK API
from robolink import *    # Robot toolbox
import time
import tkinter as tk
from tkinter import simpledialog, messagebox

# Initialize Tkinter root
root = tk.Tk()
root.withdraw()  # Hide the root window

# Get the current working directory (where the RoboDK file should be)
current_directory = os.getcwd()
log_file = os.path.join(current_directory, "robot_log.txt")

# Function to log messages
def log(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

# Initialize the log file (this will overwrite the old file)
with open(log_file, "w") as f:
    f.write("Robot log started\n")

# Ask user for the starting sanding point
start_point = simpledialog.askinteger("Input", "Enter the starting sanding point (1-50):")

# Verify the input
if start_point is None:
    messagebox.showerror("Error", "No input provided. Exiting the program.")
    exit()
elif start_point < 1 or start_point > 50:
    messagebox.showerror("Error", "Starting point must be between 1 and 50. Exiting the program.")
    exit()

# Ask user for the movements
movements_input = simpledialog.askstring("Input", "Enter movements as a comma-separated list (e.g., 0,1,2,3,4):")

# Verify the input
if not movements_input:
    messagebox.showerror("Error", "No movements provided. Exiting the program.")
    exit()

# Parse the movements
try:
    movements = list(map(int, movements_input.split(',')))
except ValueError:
    messagebox.showerror("Error", "Invalid input format for movements. Exiting the program.")
    exit()

# Link to RoboDK
RL = Robolink()

# Notify user:
log('To edit this program:\nright click on the Python program, then, select "Edit Python script"')

# List all items in the station
all_items = RL.ItemList()
log("Available items in the station:")
for item in all_items:
    log(item.Name())

# Verify and get the robot item
robot = RL.Item('UR30')
if not robot.Valid():
    raise Exception("Robot 'UR30' not found")
else:
    log("Robot 'UR30' found")

# Verify and get the 'Home' item
home = RL.Item('Home')
if not home.Valid():
    raise Exception("Target 'Home' not found")
else:
    log("Target 'Home' found")

# Move robot to home
log("Moving to 'Home'")
robot.MoveL(home)

# Define the grid layout
grid = [
    [10, 11, 30, 31, 50],
    [9, 12, 29, 32, 49],
    [8, 13, 28, 33, 48],
    [7, 14, 27, 34, 47],
    [6, 15, 26, 35, 46],
    [5, 16, 25, 36, 45],
    [4, 17, 24, 37, 44],
    [3, 18, 23, 38, 43],
    [2, 19, 22, 39, 42],
    [1, 20, 21, 40, 41]
]

# Find the initial coordinates in the grid
initial_x = initial_y = None
for y in range(len(grid)):
    if start_point in grid[y]:
        initial_y = y
        initial_x = grid[y].index(start_point)
        break

if initial_x is None or initial_y is None:
    raise Exception(f"Starting point '{start_point}' not found in the grid")

# Move robot to the starting sanding point
initial_target_name = f'SandingPoint {start_point}'
initial_target = RL.Item(initial_target_name)
if not initial_target.Valid():
    raise Exception(f"Initial target '{initial_target_name}' not found")
else:
    log(f"Moving to initial target '{initial_target_name}'")
    robot.MoveL(initial_target)
    time.sleep(1)

# Helper function to move to the next point based on direction
def move_to_next_point(x, y, direction):
    if direction == 1:  # up
        return x, y - 1
    elif direction == 2:  # down
        return x, y + 1
    elif direction == 3:  # right
        return x + 1, y
    elif direction == 4:  # left
        return x - 1, y
    elif direction == 0:  # stay
        return x, y

# Follow the path based on user movements
current_x, current_y = initial_x, initial_y
for move in movements:
    next_x, next_y = move_to_next_point(current_x, current_y, move)
    if move != 0:  # Skip movement if direction is 0 (stay)
        if 0 <= next_x < len(grid[0]) and 0 <= next_y < len(grid):
            next_point_index = grid[next_y][next_x]
            next_target_name = f'SandingPoint {next_point_index}'
            next_target = RL.Item(next_target_name)
            if not next_target.Valid():
                log(f"Next target '{next_target_name}' not found. Skipping this movement.")
                continue
            else:
                target_position = next_target.Pose().Pos()
                log(f"Moving to '{next_target_name}' at grid coordinates ({next_x}, {next_y}) with position {target_position}")
                try:
                    robot.MoveL(next_target)
                    time.sleep(1)  # stay at the sanding point for 1 second
                except Exception as e:
                    log(f"Error moving to target '{next_target_name}': {e}")
                    continue
            current_x, current_y = next_x, next_y
        else:
            log(f"Invalid move '{move}' from ({current_x}, {current_y}). Skipping this movement.")
    else:
        log(f"Staying at '{current_x}, {current_y}' for 1 second")
        time.sleep(1)  # stay at the current position for 1 second

# Move robot back to home after completing the path
log("Returning to 'Home'")
robot.MoveL(home)