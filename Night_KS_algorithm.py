
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm

# Function to enhance images using a proposed algorithm
def enhance_image(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Normalize the V channel
    norm_V = hsv[:,:,2] / 255.0

    # Perform decomposition
    L = np.exp(-1 * np.log(norm_V + 0.001))
    R = norm_V / L

    # Adjust illumination
    L_adjusted = cv2.normalize(L, None, alpha=0.5, beta=400, norm_type=cv2.NORM_MINMAX)

    # Generate enhanced V channel
    enhanced_V = L_adjusted * R
    enhanced_V = cv2.normalize(enhanced_V, None, alpha=0, beta=1500, norm_type=cv2.NORM_MINMAX)
    enhanced_V = enhanced_V.astype(np.uint8)

    # Merge enhanced V channel with original H and S channels
    enhanced_hsv = cv2.merge((hsv[:,:,0], hsv[:,:,1], enhanced_V))

    # Convert enhanced HSV image back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
  
    return enhanced_image

# Function to calculate Peak Signal-to-Noise Ratio (PSNR)
def calculate_psnr(img1, img2):
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)

    # Handle case where images are identical
    if mse == 0:
        return float('inf')

    # Calculate PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    return psnr

# Path to the folder containing images
input_folder = r"C:\Users\91981\Desktop\OnePlus_Photo\Night_images"

# Get list of image files in the folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_images = len(image_files)

# Use tqdm to create a progress bar
with tqdm(total=total_images, desc="Processing images") as pbar:
    psnr_values = []
    ssim_values = []

    for i, file_name in enumerate(image_files):
        # Load and resize image
        image_path = os.path.join(input_folder, file_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (612, 812))

        # Enhance image
        enhanced_image = enhance_image(image)

        # Save enhanced image
        output_folder = r'C:\Users\91981\Desktop\OnePlus_Photo\Enhanced_Night_Images'
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, enhanced_image)

        # Calculate PSNR
        psnr_value = calculate_psnr(image, enhanced_image)
        psnr_values.append(psnr_value)

        # Calculate SSIM
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced_image_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        ssim_index = ssim(image_gray, enhanced_image_gray, win_size=5)
        ssim_values.append(ssim_index)

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix_str("{:.1f}%".format((i+1) / total_images * 100))


        


