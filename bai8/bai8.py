import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh vệ tinh
img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)

# Áp dụng bộ lọc Gaussian
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

# Phát hiện cạnh sử dụng toán tử Sobel
sobelx = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.bitwise_or(sobelx, sobely)

# Phát hiện cạnh sử dụng toán tử Prewitt
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
prewittx = cv2.filter2D(blurred_img, -1, kernelx)
prewitty = cv2.filter2D(blurred_img, -1, kernely)
prewitt_combined = cv2.bitwise_or(prewittx, prewitty)

# Phát hiện cạnh sử dụng toán tử Robert
roberts_cross_v = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_cross_h = np.array([[0, 1], [-1, 0]], dtype=np.float32)
robertsx = cv2.filter2D(blurred_img, -1, roberts_cross_v)
robertsy = cv2.filter2D(blurred_img, -1, roberts_cross_h)
roberts_combined = cv2.bitwise_or(robertsx, robertsy)

# Phát hiện cạnh sử dụng toán tử Canny
edges = cv2.Canny(blurred_img, 100, 200)

# Hiển thị kết quả
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.axis('off')

plt.subplot(2, 3, 2), plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel'), plt.axis('off')

plt.subplot(2, 3, 3), plt.imshow(prewitt_combined, cmap='gray')
plt.title('Prewitt'), plt.axis('off')

plt.subplot(2, 3, 4), plt.imshow(roberts_combined, cmap='gray')
plt.title('Roberts'), plt.axis('off')

plt.subplot(2, 3, 5), plt.imshow(edges, cmap='gray')
plt.title('Canny'), plt.axis('off')

plt.show()
