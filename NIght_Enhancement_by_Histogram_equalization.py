
import cv2
import numpy as np
import os
from tqdm import tqdm

# Path to the folder containing images
input_dir = r"C:\Users\91981\Desktop\OnePlus_Photo\Night_images"

# Get a list of all the image files in the folder
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_files = len(image_files)

# Define a function to enhance night images
def enhance_night_photos():
    with tqdm(total=total_files, desc="Enhancing images") as progress_bar:
        for idx, filename in enumerate(image_files):
            # Load the image
            file_path = os.path.join(input_dir, filename)
            original_img = cv2.imread(file_path)
            resized_img = cv2.resize(original_img, (812, 612))

            # Apply bilateral filter
            filtered_img = cv2.bilateralFilter(resized_img, 2, 8, 8)
            # Sharpen the image
            blurred_img = cv2.GaussianBlur(filtered_img, (3, 3), 2)
            sharpened_img = cv2.addWeighted(filtered_img, 1, blurred_img, -0.5, 0)

            # Adjust brightness and contrast
            brightness_mat = np.ones(sharpened_img.shape, dtype="uint8") * 6
            contrast_mat = np.ones(sharpened_img.shape) * 1.5
            bright_img = cv2.add(sharpened_img, brightness_mat)
            bright_contrast_img = np.uint8(np.clip(cv2.multiply(np.float64(bright_img), contrast_mat), 0, 255))

            # Histogram equalization function
            def histogram_equalization(image):
                channels = cv2.split(image)
                equalized_channels = [cv2.equalizeHist(channel) for channel in channels]
                equalized_img = cv2.merge(equalized_channels)
                return equalized_img

            # Apply histogram equalization
            equalized_img = histogram_equalization(bright_contrast_img)
            diff_img = cv2.absdiff(bright_contrast_img, sharpened_img)
            enhanced_img = cv2.add(6 * diff_img, sharpened_img)

            # Adaptive thresholding
            gray_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
            adaptive_thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            adaptive_thresh = cv2.bitwise_not(adaptive_thresh)
            masked_img = cv2.add(enhanced_img, cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR))
            masked_img = cv2.bitwise_not(masked_img)

            # Saturation adjustment
            hsv_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
            hsv_img[..., 1] = np.uint8(np.clip(hsv_img[..., 1] * 1.7, 20, 150))
            final_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

            output_dir = r'C:\Users\91981\Desktop\OnePlus_Photo\Night_images_enhanced'
            output_path = os.path.join(output_dir, filename)

            # Save the enhanced image
            cv2.imwrite(output_path, final_img)
            progress_bar.update(1)
            progress_bar.set_postfix_str("{:.1f}%".format((idx + 1) / total_files * 100))

# Call the function to enhance low light images
enhance_night_photos()
