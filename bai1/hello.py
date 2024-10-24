import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to apply negative transformation
def negative_image(image):
    return 255 - image

# Function to adjust image contrast using histogram equalization
def enhance_contrast(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    else:  # Color image
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# Function to apply log transformation
def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    return np.array(log_image, dtype=np.uint8)

# Function to apply histogram equalization
def histogram_equalization(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    else:  # Color image
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# Load the image
image_path = 'header4a.jpg'  # Change to your image path
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convert to grayscale for processing if needed
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply transformations
negative_img = negative_image(image)
contrast_img = enhance_contrast(image)
log_img = log_transform(gray_image)
hist_eq_img = histogram_equalization(image)

# Display results using matplotlib
titles = ['Original Image', 'Negative Image', 'Contrast Enhanced', 'Log Transformed', 'Histogram Equalized']
images = [image, negative_img, contrast_img, log_img, hist_eq_img]

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
