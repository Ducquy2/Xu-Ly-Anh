import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from fcmeans import FCM
from sklearn.datasets import load_iris

# Hàm đánh giá phân cụm
def evaluate_clustering(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    rand_index = adjusted_rand_score(y_true, y_pred)
    return f1, rand_index

# Thực hiện phân cụm trên bộ dữ liệu IRIS
def cluster_iris():
    # Tải dữ liệu IRIS
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target

    # K-means
    kmeans_iris = KMeans(n_clusters=3, random_state=42).fit(X_iris)
    labels_kmeans_iris = kmeans_iris.labels_

    # Fuzzy C-Means (FCM)
    fcm_iris = FCM(n_clusters=3, random_state=42)
    fcm_iris.fit(X_iris)
    labels_fcm_iris = fcm_iris.predict(X_iris)

    # Agglomerative Hierarchical Clustering (AHC)
    ahc_iris = AgglomerativeClustering(n_clusters=3)
    labels_ahc_iris = ahc_iris.fit_predict(X_iris)

    # Đánh giá K-means
    kmeans_f1_iris, kmeans_rand_iris = evaluate_clustering(y_iris, labels_kmeans_iris)
    # Đánh giá FCM
    fcm_f1_iris, fcm_rand_iris = evaluate_clustering(y_iris, labels_fcm_iris)
    # Đánh giá AHC
    ahc_f1_iris, ahc_rand_iris = evaluate_clustering(y_iris, labels_ahc_iris)

    print(f"K-means (IRIS): F1-score = {kmeans_f1_iris}, RAND index = {kmeans_rand_iris}")
    print(f"FCM (IRIS): F1-score = {fcm_f1_iris}, RAND index = {fcm_rand_iris}")
    print(f"AHC (IRIS): F1-score = {ahc_f1_iris}, RAND index = {ahc_rand_iris}")

# Thực hiện phân cụm trên ảnh giao thông (ảnh vệ tinh)
def cluster_traffic_image(image_path):
    # Đọc ảnh và chuyển thành mảng dữ liệu
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)

    # K-means
    kmeans_traffic = KMeans(n_clusters=3, random_state=42).fit(pixels)
    labels_kmeans_traffic = kmeans_traffic.labels_
    kmeans_segmented_image = kmeans_traffic.cluster_centers_[labels_kmeans_traffic].reshape(image.shape)

    # Fuzzy C-Means (FCM)
    fcm_traffic = FCM(n_clusters=3, random_state=42)
    fcm_traffic.fit(pixels)
    labels_fcm_traffic = fcm_traffic.predict(pixels)
    fcm_segmented_image = fcm_traffic.centers_[labels_fcm_traffic].reshape(image.shape)

    # Agglomerative Hierarchical Clustering (AHC)
    ahc_traffic = AgglomerativeClustering(n_clusters=3)
    labels_ahc_traffic = ahc_traffic.fit_predict(pixels)
    ahc_segmented_image = labels_ahc_traffic.reshape(image.shape[0], image.shape[1], 3) * 85  # Scale labels

    # Hiển thị kết quả
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original Image')
    axs[0, 1].imshow(kmeans_segmented_image.astype(int))
    axs[0, 1].set_title('K-means')
    axs[1, 0].imshow(fcm_segmented_image.astype(int))
    axs[1, 0].set_title('FCM')
    axs[1, 1].imshow(ahc_segmented_image.astype(int))
    axs[1, 1].set_title('AHC')
    plt.show()

# Chạy phân cụm trên bộ dữ liệu IRIS
cluster_iris()

# Chạy phân cụm trên ảnh giao thông
# Đường dẫn tới ảnh giao thông
image_path = 'img.png'
cluster_traffic_image(image_path)
