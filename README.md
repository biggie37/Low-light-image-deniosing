# Low-light-image-deniosing
# Low-Light Image Enhancement Project

## Overview

This project focuses on enhancing low-light images using various image processing techniques implemented in Python using OpenCV and NumPy libraries. The goal is to improve the visibility and quality of images captured under low-light conditions.

## Project Structure

### Script 1: Basic Enhancement
- **Filename:** `basic_enhancement.py`
- **Description:** Implements basic image enhancement techniques including bilateral filtering, sharpening, brightness and contrast adjustments, saturation enhancement, and denoising using Non-Local Means.
- **Usage:** Place your low-light images in a folder specified in the script, then run the script to enhance them.

### Script 2: Nature-Inspired Enhancement
- **Filename:** `nature_inspired_enhancement.py`
- **Description:** Applies nature-inspired image enhancement techniques such as histogram equalization and gamma correction in addition to basic techniques like bilateral filtering and sharpening.
- **Usage:** Similar to Script 1, place your low-light images in a folder and run the script to enhance them.

### Script 3: Sharp Image Enhancement
- **Filename:** `sharp_image_enhancement.py`
- **Description:** Focuses on sharpening low-light images using bilateral filtering, Gaussian blur, and adjustment of brightness and contrast based on the average pixel value of the image.
- **Usage:** Place your low-light images in a folder and run the script to apply the sharp image enhancement techniques.

### Script 4: Comprehensive Enhancement
- **Filename:** `comprehensive_enhancement.py`
- **Description:** Combines various enhancement techniques including bilateral filtering, sharpening, brightness and contrast adjustments, saturation enhancement, and denoising using Non-Local Means.
- **Usage:** Similar to previous scripts, place your low-light images in a folder and run the script to enhance them comprehensively.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- tqdm (for progress bars)
- scikit-image (for SSIM metric, if needed)

## Setup and Usage

1. **Installation:**
   - Ensure Python 3.x is installed on your system.
   - Install required libraries using pip:
     ```
     pip install opencv-python numpy tqdm scikit-image
     ```

2. **Execution:**
   - Place your low-light images in the specified input folder (`Night_images`).
   - Run the desired script (e.g., `python basic_enhancement.py`) to process and enhance the images.
   - Enhanced images will be saved in the `Night_images_enhanced` folder.

3. **Adjustments:**
   - Modify parameters and adjustments within the scripts as needed, such as image resizing, filtering parameters, or enhancement algorithms.

## Notes

- These scripts assume images are in common formats like PNG, JPEG, BMP, etc., and located in the specified input folder.
- The enhancement techniques applied may vary based on the characteristics and darkness levels of the input images.
- Consider experimenting with different parameters and techniques for optimal results based on your specific use case.

## Credits

- Developed by [Nischay Vermani]

