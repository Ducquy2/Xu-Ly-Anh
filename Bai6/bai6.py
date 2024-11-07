import numpy as np
import cv2
import matplotlib.pyplot as plt

# Bước 1: Đọc ảnh và chuyển đổi thành mảng dữ liệu
image = cv2.imread('nha1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = image.reshape(-1, 3)

# Chuẩn hóa dữ liệu
pixels = np.float32(pixels) / 255.0

def initialize_centroids(pixels, k):
    indices = np.random.choice(pixels.shape[0], k, replace=False)
    return pixels[indices]

def compute_distance(pixel, centroid):
    return np.linalg.norm(pixel - centroid)

def assign_clusters(pixels, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for pixel in pixels:
        distances = [compute_distance(pixel, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(pixel)
    return clusters

def compute_centroids(clusters):
    return [np.mean(cluster, axis=0) for cluster in clusters]

def has_converged(old_centroids, new_centroids, tol=1e-4):
    return np.all([compute_distance(old, new) < tol for old, new in zip(old_centroids, new_centroids)])

def kmeans(pixels, k, max_iters=100):
    centroids = initialize_centroids(pixels, k)
    for _ in range(max_iters):
        clusters = assign_clusters(pixels, centroids)
        new_centroids = compute_centroids(clusters)
        if has_converged(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

def show_clustered_image(pixels, clusters, centroids):
    pixels_new = np.zeros_like(pixels)
    for i, cluster in enumerate(clusters):
        for pixel in cluster:
            index = np.where((pixels == pixel).all(axis=1))[0][0]
            pixels_new[index] = centroids[i]
    clustered_image = pixels_new.reshape(image.shape)
    plt.imshow(clustered_image)
    plt.axis('off')
    plt.show()

# Bước 2-5: Thực hiện K-means với các giá trị K khác nhau
for k in [2, 3, 4, 5]:
    centroids, clusters = kmeans(pixels, k)
    show_clustered_image(pixels, clusters, centroids)
