model_path="model.keras"
#---------------------------------
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import keras
import numpy as np
import tensorflow as tf

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Wykrywanie deepfake'ów")
        self.setGeometry(300,300, 440, 305)

        self.model = keras.saving.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.setStyleSheet("background-color: linen;")
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.layout = QGridLayout()
        self.centralWidget.setLayout(self.layout)

        self.load_button = QPushButton("Wczytaj zdjęcie")
        self.load_button.setStyleSheet("background-color: burlywood; color: black;")
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button, 0, 0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label, 2, 0)

        self.result_label = QLabel()
        self.layout.addWidget(self.result_label, 1, 0)

    def load_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Wybierz zdjęcie", "", "Obrazy (*.png *.jpg *.jpeg)")
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.resize(image, [128, 128])
            image_array = np.expand_dims(image.numpy(), axis=0)
            prediction = self.model.predict(image_array)
            deepfake_score = float(prediction[0][0])
        except Exception as e:
            print("Błąd przewidywania obrazu:", e)
        deepfake_score = np.round(deepfake_score,4)*100
        self.result_label.setText(f"Prawdopodobieństwo na to, że zdjęcie jest prawdziwe: {deepfake_score}%")
        #
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaledToWidth(200)
        self.image_label.setPixmap(scaled_pixmap)
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())