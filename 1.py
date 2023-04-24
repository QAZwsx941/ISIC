import os
import numpy as np
from PIL import Image

# Define the input and output paths
input_folder = 'imgs'
output_folder = 'imgs'

# Get a list of all the .npy files in the input folder
npy_files = [file for file in os.listdir(input_folder) if file.endswith('.npy')]

# Loop over each .npy file, load it, convert it to a PNG image, and save it to the output folder
for npy_file in npy_files:
    # Load the .npy file using numpy
    data = np.load(os.path.join(input_folder, npy_file))

    # Convert the numpy array to a PIL Image object
    image = Image.fromarray(data)

    # Save the image as a PNG file in the output folder
    image.save(os.path.join(output_folder, os.path.splitext(npy_file)[0] + '.png'))
