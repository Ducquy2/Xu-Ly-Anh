import sys
import os
import joblib
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import precision_score, recall_score, accuracy_score

from pythonProject.bai4.bai4 import X_test, y_test


class ResultWindow(QWidget):
    def __init__(self, results):
        super().__init__()
        self.setWindowTitle("Kết quả huấn luyện")
        layout = QVBoxLayout()

        for model_name, metrics in results.items():
            label = QLabel(
                f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            layout.addWidget(label)

        self.setLayout(layout)


class PlotWindow(QWidget):
    def __init__(self, results):
        super().__init__()
        self.setWindowTitle("Đồ thị kết quả")
        layout = QVBoxLayout()

        fig = Figure()
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        ax = fig.add_subplot(111)
        model_names = list(results.keys())
        accuracy = [results[m]['accuracy'] for m in model_names]
        precision = [results[m]['precision'] for m in model_names]
        recall = [results[m]['recall'] for m in model_names]

        x = np.arange(len(model_names))
        ax.bar(x - 0.2, accuracy, 0.2, label='Accuracy')
        ax.bar(x, precision, 0.2, label='Precision')
        ax.bar(x + 0.2, recall, 0.2, label='Recall')

        ax.set_xlabel('Model')
        ax.set_title('Kết quả mô hình')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()

        self.setLayout(layout)
        self.canvas.draw()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phân loại ảnh")
        self.setGeometry(100, 100, 400, 300)
        layout = QVBoxLayout()

        self.train_button = QPushButton("Hiển thị kết quả huấn luyện", self)
        self.train_button.clicked.connect(self.show_results)
        layout.addWidget(self.train_button)

        self.plot_button = QPushButton("Hiển thị đồ thị kết quả", self)
        self.plot_button.clicked.connect(self.show_plot)
        layout.addWidget(self.plot_button)

        self.combo_box = QComboBox(self)
        self.combo_box.addItems(["SVM", "KNN", "Decision Tree"])
        layout.addWidget(self.combo_box)

        self.select_button = QPushButton("Chọn ảnh để phân loại", self)
        self.select_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_button)

        self.result_label = QLabel("", self)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def show_results(self):
        results = self.load_models()
        self.result_window = ResultWindow(results)
        self.result_window.show()

    def show_plot(self):
        results = self.load_models()
        self.plot_window = PlotWindow(results)
        self.plot_window.show()

    def load_models(self):
        models = ["SVM", "KNN", "Decision Tree"]
        results = {}
        for model_name in models:
            model = joblib.load(f'{model_name}_model.joblib')
            # Giả lập kết quả cho hiển thị đồ thị, thực tế sẽ sử dụng kết quả đã lưu
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
        return results

    def select_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "",
                                                   "Image Files (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_path:
            try:
                model_name = self.combo_box.currentText()
                model = joblib.load(f'{model_name}_model.joblib')
                img = cv2.imread(file_path)
                if img is None:
                    QMessageBox.warning(self, "Lỗi", "Không thể đọc ảnh. Vui lòng chọn ảnh hợp lệ.")
                    return

                img = cv2.resize(img, (128, 128)).flatten().reshape(1, -1)
                prediction = model.predict(img)
                self.result_label.setText(f"Kết quả phân loại: {prediction[0]}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Đã xảy ra lỗi: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
