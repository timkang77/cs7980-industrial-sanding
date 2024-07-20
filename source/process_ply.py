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

def create_3d_matrix(x, y, z, min_x, min_y, x_interval, y_interval, num_cols, num_rows):
    """Create a 3D array with the mean, frequency, and min z values."""
    matrix = np.full((num_cols, num_rows, 3), np.nan)  # Initialize 3D array for (mean, frequency, min_value)
    sum_matrix = np.zeros((num_cols, num_rows))
    count_matrix = np.zeros((num_cols, num_rows))
    min_matrix = np.full((num_cols, num_rows), np.inf)

    for xi, yi, zi in zip(x, y, z):
        col = int((xi - min_x) / x_interval)
        row = int((yi - min_y) / y_interval)
        
        # Ensure the indices are within bounds
        if col >= num_cols:
            col = num_cols - 1
        if row >= num_rows:
            row = num_rows - 1
        
        if 0 <= col < num_cols and 0 <= row < num_rows:
            sum_matrix[col, row] += zi
            count_matrix[col, row] += 1
            if zi < min_matrix[col, row]:
                min_matrix[col, row] = zi

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_matrix = np.divide(sum_matrix, count_matrix)
        mean_matrix[count_matrix == 0] = np.nan

    for col in range(num_cols):
        for row in range(num_rows):
            matrix[col, row, 0] = mean_matrix[col, row]
            matrix[col, row, 1] = count_matrix[col, row]
            matrix[col, row, 2] = min_matrix[col, row] if count_matrix[col, row] > 0 else np.nan

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

def global_variance(file_path):
    """Calculate the global variance of the z values."""
    x, y, z = load_ply(file_path)
    mean_z = np.mean(z)
    variance_z = np.mean((z - mean_z) ** 2)
    return variance_z

def output_grid(file_path, num_cols, num_rows):
    x, y, z = load_ply(file_path)
    min_x, max_x, min_y, max_y, x_interval, y_interval = compute_intervals(x, y, num_cols, num_rows)
    matrix = create_3d_matrix(x, y, z, min_x, min_y, x_interval, y_interval, num_cols, num_rows)
    """
    print(f'x range: {min_x} - {max_x}')
    print(f'y range: {min_y} - {max_y}')

    # Print the 3D matrix
    print("3D Matrix with (mean, frequency, min_value):")
    for row in matrix:
        print(row)
    
    plot_scatter(x, y, z)"""
    return matrix