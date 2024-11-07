import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time

app = Flask(__name__)


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


@app.route('/train', methods=['POST'])
def train_models():
    folder_path = request.json['E:/JetBrains/Python/pythonProject/Bai3/Anh']
    images, labels = load_images_from_folder(folder_path)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    models = {
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'Decision Tree': DecisionTreeClassifier()
    }

    results = {}
    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = time.time()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=1)

        results[model_name] = {
            'time': end_time - start_time,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
