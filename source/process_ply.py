import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import csv

def load_ply(file_path):
    """Load the PLY file and return the vertex data."""
    plydata = PlyData.read(file_path)
    x = plydata.elements[0].data['x']
    y = plydata.elements[0].data['y']
    z = plydata.elements[0].data['z']
    return x, y, z

def compute_intervals(x, y, num_cols, num_rows):
    """Compute the intervals for x and y axes."""
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    x_interval = (max_x - min_x) / num_cols
    y_interval = (max_y - min_y) / num_rows
    return min_x, max_x, min_y, max_y, x_interval, y_interval

def create_2d_matrix(x, y, z, min_x, min_y, x_interval, y_interval, num_cols, num_rows):
    """Create a 2D matrix with the mean z values."""
    matrix = np.full((num_rows, num_cols), np.nan)
    count_matrix = np.zeros((num_rows, num_cols))
    points_matrix = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
    x_ranges = [[(np.inf, -np.inf) for _ in range(num_cols)] for _ in range(num_rows)]
    y_ranges = [[(np.inf, -np.inf) for _ in range(num_cols)] for _ in range(num_rows)]

    for xi, yi, zi in zip(x, y, z):
        col = int((xi - min_x) / x_interval)
        row = int((yi - min_y) / y_interval)
        
        if 0 <= col < num_cols and 0 <= row < num_rows:
            points_matrix[row][col].append(zi)
            if np.isnan(matrix[row, col]):
                matrix[row, col] = 0.0
            matrix[row, col] += zi
            count_matrix[row, col] += 1

            # Update x and y ranges
            x_min, x_max = x_ranges[row][col]
            y_min, y_max = y_ranges[row][col]
            x_ranges[row][col] = (float(min(xi, x_min)), float(max(xi, x_max)))
            y_ranges[row][col] = (float(min(yi, y_min)), float(max(yi, y_max)))

    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.divide(matrix, count_matrix)
        matrix[count_matrix == 0] = np.nan

    return matrix, points_matrix, count_matrix, x_ranges, y_ranges

def plot_scatter(x, y, z):
    """Plot a scatter plot of the vertex data."""
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(x, y, c=z, cmap='viridis', marker='o')
    plt.colorbar(sc, label='Z value')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Scatter Plot of Vertices with Z value as color')
    plt.show()

def save_to_csv(file_path, num_cols, num_rows, points_matrix, count_matrix, x_ranges, y_ranges):
    """Save the grid information to a CSV file."""
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Grid Group [x, y]", "x range", "y range", "Point Clouds (e.g.)", "Point Cloud Count"])
        
        for row in range(num_rows):
            for col in range(num_cols):
                x_range = x_ranges[row][col]
                y_range = y_ranges[row][col]
                point_clouds = points_matrix[row][col]
                point_cloud_count = int(count_matrix[row][col])
                writer.writerow([f"[{row + 1}, {col + 1}]", x_range, y_range, point_clouds, point_cloud_count])

def main(file_path, num_cols, num_rows, csv_output_path):
    x, y, z = load_ply(file_path)
    min_x, max_x, min_y, max_y, x_interval, y_interval = compute_intervals(x, y, num_cols, num_rows)
    matrix, points_matrix, count_matrix, x_ranges, y_ranges = create_2d_matrix(x, y, z, min_x, min_y, x_interval, y_interval, num_cols, num_rows)
    print(f'x range: {min_x} - {max_x}')
    print(f'y range: {min_y} - {max_y}')

    # Print the 2D matrix
    matrix_2d_list = matrix.tolist()
    print("2D Matrix with mean z values:")
    for row in matrix_2d_list:
        print(row)
    
    plot_scatter(x, y, z)
    save_to_csv(csv_output_path, num_cols, num_rows, points_matrix, count_matrix, x_ranges, y_ranges)

# Parameters
file_path = "/Users/chengyan/Desktop/7980/cs7980-industrial-sanding/simulation/RoboDK/TableTop_half_randomized.ply"
csv_output_path = "/Users/chengyan/Desktop/7980/cs7980-industrial-sanding/source/output.csv"
num_cols = 10
num_rows = 5

# Run the main function
main(file_path, num_cols, num_rows, csv_output_path)