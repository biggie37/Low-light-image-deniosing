import cv2
import os
from tqdm import tqdm

# Define source and destination directories
source_directory = r'C:\Users\SOHAM PADHYE\Documents\Sharp_blur_dataset\CV_project\Input\Night_KB'
destination_directory = r'C:\Users\SOHAM PADHYE\Documents\Sharp_blur_dataset\CV_project\output\Blurred_images'

# Ensure the destination directory exists; create if necessary
os.makedirs(destination_directory, exist_ok=True)

# Get list of all image files in the source directory
image_files = [filename for filename in os.listdir(source_directory) if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# Process each image with Gaussian blur
for filename in tqdm(image_files, desc="Processing images"):
    try:
        # Read the image
        image_path = os.path.join(source_directory, filename)
        image = cv2.imread(image_path)

        # Apply Gaussian blur
        kernel_size = (121, 11)  # should be odd
        blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

        # Save the blurred image to the destination directory
        output_path = os.path.join(destination_directory, filename)
        cv2.imwrite(output_path, blurred_image)
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

print('DONE')
