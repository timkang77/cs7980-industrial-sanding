import random
import os

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            columns = line.strip().split()
            if len(columns) != 3:
                continue  # Skip lines that do not have exactly 3 columns
            columns[2] = str(random.gauss(0.15, 0.01))
            outfile.write(' '.join(columns) + '\n')

input_file = os.path.join(os.path.dirname(__file__), 'input.txt')  # Relative path
output_file = os.path.join(os.path.dirname(__file__), 'output.txt')  # Relative path
process_file(input_file, output_file)