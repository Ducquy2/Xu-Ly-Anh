import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread('header4a.jpg', cv2.IMREAD_GRAYSCALE)

# Toán tử Sobel thủ công
sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Áp dụng Sobel
grad_x = cv2.filter2D(image, cv2.CV_64F, sobelx)
grad_y = cv2.filter2D(image, cv2.CV_64F, sobely)
sobel_combined = cv2.magnitude(grad_x, grad_y)

# Toán tử Laplacian of Gaussian (LoG) thủ công
log_kernel = np.array([[0, 0, -1, 0, 0],
                       [0, -1, -2, -1, 0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0]])

# Áp dụng LoG
log = cv2.filter2D(image, cv2.CV_64F, log_kernel)

# Hiển thị kết quả
plt.figure(figsize=(15, 8))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Hình ảnh gốc')
plt.subplot(1, 3, 2), plt.imshow(sobel_combined, cmap='gray'), plt.title('Phát hiện cạnh Sobel')
plt.subplot(1, 3, 3), plt.imshow(log, cmap='gray'), plt.title('Laplacian của Gaussian (LoG)')
plt.show()
