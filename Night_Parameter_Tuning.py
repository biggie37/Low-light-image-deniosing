
import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# Path to the folder containing input images
input_folder = r"C:\Users\91981\Desktop\OnePlus_Photo\Night_images"

# Get a list of all the image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_images = len(image_files)

def enhance_images():
    with tqdm(total=total_images, desc="Processing images") as pbar:
        for i, filename in enumerate(image_files):
            # Load the image
            file_path = os.path.join(input_folder, filename)
            original_img = cv2.imread(file_path)
            original_img = cv2.resize(original_img, (812, 612))
            
            # Apply bilateral filtering
            smoothed_img = cv2.bilateralFilter(original_img, 2, 10, 10)

            # Sharpen the image
            kernel = np.array([[0, -5, 0], [-5, 8, -5], [0, -5, 0]])
            blurred_img = cv2.GaussianBlur(smoothed_img, (5, 5), 2)
            sharpened_img = cv2.addWeighted(smoothed_img, 1, blurred_img, -0.5, 0)

            # Calculate average pixel value for brightness adjustment
            avg_pixel_value = np.mean(original_img)
            
            # Adjust brightness and contrast based on average pixel value
            if avg_pixel_value < 10:
                brightness_matrix = np.ones(original_img.shape, dtype="uint8") * 5
                contrast_matrix = np.ones(original_img.shape) * 3
            elif 10 <= avg_pixel_value < 20:
                brightness_matrix = np.ones(original_img.shape, dtype="uint8") * 3
                contrast_matrix = np.ones(original_img.shape) * 2
            else:
                brightness_matrix = np.ones(original_img.shape, dtype="uint8") * 2
                contrast_matrix = np.ones(original_img.shape) * 2
            
            brightened_img = cv2.add(sharpened_img, brightness_matrix)
            contrasted_img = np.uint8(np.clip(cv2.multiply(np.float64(brightened_img), contrast_matrix), 0, 255))
            
            # Calculate difference and apply to enhance image
            diff_img = cv2.absdiff(contrasted_img, original_img)
            enhanced_img = cv2.add(3 * diff_img, original_img)
            
            # Convert to HSV and adjust saturation
            hsv_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
            hsv_img[..., 1] = np.uint8(np.clip(hsv_img[..., 1] * 1.3, 20, 150))
            sat_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            
            # Apply Non-Local Means Denoising
            denoised_img = cv2.fastNlMeansDenoisingColored(sat_img, None, 3, 3, 20, 15)
            
            # Save the enhanced image to the output folder
            output_folder = r'C:\Users\91981\Desktop\OnePlus_Photo\Night_images_enhanced'
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, denoised_img)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix_str("{:.1f}%".format((i + 1) / total_images * 100))

# Call the function to process images
enhance_images()

        
