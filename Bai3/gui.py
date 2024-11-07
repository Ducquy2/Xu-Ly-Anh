import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit
import requests
import json

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Phân loại hình ảnh'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)

        layout = QVBoxLayout()

        self.label = QLabel('Chọn thư mục chứa hình ảnh:', self)
        layout.addWidget(self.label)

        self.textEdit = QTextEdit(self)
        layout.addWidget(self.textEdit)

        self.button = QPushButton('Chọn thư mục', self)
        self.button.clicked.connect(self.showDialog)
        layout.addWidget(self.button)

        self.trainButton = QPushButton('Huấn luyện mô hình', self)
        self.trainButton.clicked.connect(self.train_model)
        layout.addWidget(self.trainButton)

        self.resultLabel = QLabel('', self)
        layout.addWidget(self.resultLabel)

        self.setLayout(layout)

    def showDialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Chọn thư mục')
        self.textEdit.setText(folder_path)

    def train_model(self):
        folder_path = self.textEdit.toPlainText()
        if folder_path:
            url = 'http://127.0.0.1:5000/train'
            data = {'E:/JetBrains/Python/pythonProject/Bai3/Anh': folder_path}
            response = requests.post(url, json=data)
            results = json.loads(response.text)
            self.resultLabel.setText(f"Kết quả: {results}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec())
