import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_score, recall_score
import joblib


def load_images_from_folder(folder):
    images = []
    labels = []
    label_dict = {'hoa': 0, 'dongvat': 1}
    for label in label_dict:
        path = os.path.join(folder, label)
        for filename in os.listdir(path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(path, filename))
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    images.append(img.flatten())
                    labels.append(label_dict[label])
    return np.array(images), np.array(labels)


folder_path = 'E:/JetBrains/Python/pythonProject/bai4/Anh'
images, labels = load_images_from_folder(folder_path)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Huấn luyện và lưu mô hình
models = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier()
}

results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)

    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

    # Lưu mô hình đã huấn luyện
    try:
        model_dt = joblib.load('Decision Tree_model.joblib')
    except Exception as e:
        print(f"Error loading model: {e}")

print(results)

