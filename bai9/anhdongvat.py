import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

# Tải bộ dữ liệu CIFAR-10 từ Keras
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Hiển thị hình ảnh gốc từ CIFAR-10
def display_image(img, label, title="Image"):
    plt.imshow(img)
    plt.title(f"{title} - Label: {label}")
    plt.axis('off')
    plt.show()

# Hiển thị một ảnh gốc từ tập huấn luyện
display_image(X_train[0], y_train[0], title="Original Image")

# Chuẩn hóa ảnh (chia giá trị pixel trong phạm vi [0, 1])
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Tiền xử lý ảnh: Chuyển ảnh về kích thước nhỏ hơn (32x32) và chuyển thành ảnh xám
def preprocess_image(img):
    # Làm mờ ảnh (Blur)
    img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])  # Chuyển ảnh thành grayscale
    img_resized = np.resize(img_gray, (32, 32))  # Thay đổi kích thước ảnh
    return img_resized

# Hiển thị hình ảnh đã qua xử lý
processed_image = preprocess_image(X_train[0])
plt.imshow(processed_image, cmap='gray')
plt.title("Processed Image")
plt.axis('off')
plt.show()

# Chuyển đổi ảnh thành một vector 1D cho các thuật toán KNN và SVM
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# ========================================
# 1. KNN (K-Nearest Neighbors)
# ========================================
# Khởi tạo và huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Đo thời gian huấn luyện
start_time = time.time()
knn.fit(X_train_flat, y_train.flatten())  # Chuyển đổi y_train thành mảng 1D
train_time_knn = time.time() - start_time

# Dự đoán và tính độ chính xác
start_time = time.time()
y_pred_knn = knn.predict(X_test_flat)
predict_time_knn = time.time() - start_time

# Đánh giá độ chính xác
accuracy_knn = accuracy_score(y_test.flatten(), y_pred_knn)

print(f"KNN - Accuracy: {accuracy_knn:.4f}, Training time: {train_time_knn:.4f}s, Prediction time: {predict_time_knn:.4f}s")

# ========================================
# 2. SVM (Support Vector Machine)
# ========================================
# Khởi tạo và huấn luyện mô hình SVM
svm = SVC(kernel='linear')

# Đo thời gian huấn luyện
start_time = time.time()
svm.fit(X_train_flat, y_train.flatten())
train_time_svm = time.time() - start_time

# Dự đoán và tính độ chính xác
start_time = time.time()
y_pred_svm = svm.predict(X_test_flat)
predict_time_svm = time.time() - start_time

# Đánh giá độ chính xác
accuracy_svm = accuracy_score(y_test.flatten(), y_pred_svm)

print(f"SVM - Accuracy: {accuracy_svm:.4f}, Training time: {train_time_svm:.4f}s, Prediction time: {predict_time_svm:.4f}s")

# ========================================
# 3. ANN (Artificial Neural Network)
# ========================================
# Xây dựng mô hình ANN
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))  # Chuyển ảnh 32x32x3 thành vector
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 lớp cho CIFAR-10

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Đo thời gian huấn luyện
start_time = time.time()
model.fit(X_train, to_categorical(y_train, 10), epochs=10, batch_size=64, verbose=0)
train_time_ann = time.time() - start_time

# Dự đoán và tính độ chính xác
start_time = time.time()
y_pred_ann = model.predict(X_test)
predict_time_ann = time.time() - start_time

# Chuyển đổi dự đoán thành nhãn (mảng một chiều)
y_pred_ann = np.argmax(y_pred_ann, axis=1)

# Đánh giá độ chính xác
accuracy_ann = accuracy_score(y_test.flatten(), y_pred_ann)

print(f"ANN - Accuracy: {accuracy_ann:.4f}, Training time: {train_time_ann:.4f}s, Prediction time: {predict_time_ann:.4f}s")

