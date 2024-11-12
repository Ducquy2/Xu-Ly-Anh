import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dữ liệu IRIS
iris = load_iris()
X = iris.data
y = iris.target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Khởi tạo các mô hình
models = {
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'SVM': SVC(kernel='linear'),
    'ANN': MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
}

# Huấn luyện và đánh giá các mô hình
results = {}
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    end_time = time.time()
    results[name] = {
        'accuracy': accuracy,
        'time': end_time - start_time
    }

# Hiển thị kết quả
for name, result in results.items():
    print(f"{name} - Accuracy: {result['accuracy']:.4f}, Time: {result['time']:.4f} seconds")

# Hiển thị hình ảnh gốc và đã qua xử lý
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Hình ảnh gốc
ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
ax[0].set_title('Original Data')

# Hình ảnh đã qua xử lý (chuẩn hóa)
ax[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=100)
ax[1].set_title('Processed Data (Standardized)')

plt.show()
