import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt

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

    for xi, yi, zi in zip(x, y, z):
        col = int((xi - min_x) / x_interval)
        row = int((yi - min_y) / y_interval)
        
        if 0 <= col < num_cols and 0 <= row < num_rows:
            if np.isnan(matrix[row, col]):
                matrix[row, col] = 0.0
            matrix[row, col] += zi
            count_matrix[row, col] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.divide(matrix, count_matrix)
        matrix[count_matrix == 0] = np.nan

    return matrix

def plot_scatter(x, y, z):
    """Plot a scatter plot of the vertex data."""
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(x, y, c=z, cmap='viridis', marker='o')
    plt.colorbar(sc, label='Z value')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Scatter Plot of Vertices with Z value as color')
    plt.show()

def main(file_path, num_cols, num_rows):
    x, y, z = load_ply(file_path)
    min_x, max_x, min_y, max_y, x_interval, y_interval = compute_intervals(x, y, num_cols, num_rows)
    matrix = create_2d_matrix(x, y, z, min_x, min_y, x_interval, y_interval, num_cols, num_rows)
    print(f'x range: {min_x} - {max_x}')
    print(f'y range: {min_y} - {max_y}')

    # Print the 2D matrix
    matrix_2d_list = matrix.tolist()
    print("2D Matrix with mean z values:")
    for row in matrix_2d_list:
        print(row)
    
    plot_scatter(x, y, z)

# Parameters
file_path = "/Users/chengyan/Desktop/7980/cs7980-industrial-sanding/simulation/RoboDK/TableTop_half_randomized.ply"
num_cols = 10
num_rows = 5

# Run the main function
main(file_path, num_cols, num_rows)
