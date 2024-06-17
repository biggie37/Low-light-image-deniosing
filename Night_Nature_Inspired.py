import cv2
import numpy as np
from tqdm import tqdm
import os
from skimage.metrics import structural_similarity as ssim

# Define the folder containing input images
input_folder = r"C:\Users\91981\Desktop\OnePlus_Photo\Night_images"

# Get a list of all the image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_images = len(image_files)

def calculate_psnr(img1, img2):
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((img1 - img2) ** 2)

    # If images are identical, PSNR is infinity
    if mse == 0:
        return float('inf')

    # Calculate PSNR using formula: PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    return psnr_value

def enhance_night_images():
    psnr_values = []
    ssim_values = []

    # Use tqdm to create a progress bar with message and percentage
    with tqdm(total=total_images, desc="Enhancing images") as pbar:
        for i, filename in enumerate(image_files):
            # Load the image
            image_path = os.path.join(input_folder, filename)
            original_image = cv2.imread(image_path)
            original_image = cv2.resize(original_image, (612, 812))
            
            # Split the image into its color channels
            blue_channel, green_channel, red_channel = cv2.split(original_image)

            # Perform nature-inspired low light enhancement on each channel
            clahe = cv2.createCLAHE(clipLimit=9.0, tileGridSize=(20, 20))
            enhanced_blue = clahe.apply(blue_channel)
            enhanced_green = clahe.apply(green_channel)
            enhanced_red = clahe.apply(red_channel)

            # Merge the enhanced channels back into an RGB image
            enhanced_image = cv2.merge((enhanced_blue, enhanced_green, enhanced_red))

            # Apply gamma correction for additional enhancement
            gamma = 1.2
            enhanced_image = np.clip((enhanced_image / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)

            # Define the output path for the enhanced image
            output_folder = r'C:\Users\91981\Desktop\OnePlus_Photo\Night_Enhanced_Nature_Inspired'
            output_path = os.path.join(output_folder, filename)
            
            # Save the enhanced image
            cv2.imwrite(output_path, enhanced_image)

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix_str("{:.1f}%".format((i + 1) / total_images * 100))

            # Calculate PSNR and SSIM
            psnr_value = calculate_psnr(original_image, enhanced_image)
            psnr_values.append(np.around(psnr_value, decimals=2))

            original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            enhanced_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
            ssim_index = ssim(original_gray, enhanced_gray, win_size=5)
            ssim_values.append(np.around(ssim_index, decimals=4))

    return psnr_values, ssim_values

# Call the function to enhance images and retrieve PSNR, SSIM values
psnr_scores, ssim_scores = enhance_night_images()

# Print the PSNR and SSIM scores for each image
for i, filename in enumerate(image_files):
    print(f"Image: {filename}, PSNR: {psnr_scores[i]}, SSIM: {ssim_scores[i]}")
