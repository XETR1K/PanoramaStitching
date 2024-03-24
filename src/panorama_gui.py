import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox
from PyQt6.QtGui import QPixmap, QImage
import cv2
from PanoramaStitcher import PanoramaStitcher
from CameraManager import CameraManager

class PanoramaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Панорамное приложение")
        self.setGeometry(100, 100, 800, 600)

        # Создание объектов PanoramaStitcher и CameraManager
        self.stitcher = PanoramaStitcher()
        self.camera_manager = CameraManager(2)

        # Создание выпадающих списков для выбора алгоритмов
        self.feature_detector_combo = QComboBox()
        self.feature_descriptor_combo = QComboBox()
        self.feature_matcher_combo = QComboBox()
        self.homography_method_combo = QComboBox()

        # Добавление алгоритмов в выпадающие списки
        self.feature_detector_combo.addItems(["SIFT", "SURF", "ORB"])
        self.feature_descriptor_combo.addItems(["BRIEF"])
        self.feature_matcher_combo.addItems(["BruteForce", "FLANN"])
        self.homography_method_combo.addItems(["RANSAC"])

        # Создание кнопки для обработки кадров
        self.process_button = QPushButton("Обработать кадры")

        # Подключение обработчика событий к кнопке
        self.process_button.clicked.connect(self.process_frames)

        # Создание макета
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Выберите алгоритмы для обнаружения ключевых точек, их описания, сопоставления и оценки гомографии:"))
        layout.addWidget(QLabel("Обнаружение ключевых точек:"))
        layout.addWidget(self.feature_detector_combo)
        layout.addWidget(QLabel("Описание ключевых точек:"))
        layout.addWidget(self.feature_descriptor_combo)
        layout.addWidget(QLabel("Сопоставление ключевых точек:"))
        layout.addWidget(self.feature_matcher_combo)
        layout.addWidget(QLabel("Оценка гомографии:"))
        layout.addWidget(self.homography_method_combo)
        layout.addWidget(self.process_button)

        # Создание виджета для размещения макета
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def process_frames(self):
        # Получение выбранных пользователем алгоритмов
        feature_detector = self.feature_detector_combo.currentText()
        feature_descriptor = self.feature_descriptor_combo.currentText()
        feature_matcher = self.feature_matcher_combo.currentText()
        homography_method = self.homography_method_combo.currentText()

        # Установка выбранных алгоритмов в PanoramaStitcher
        self.stitcher.set_feature_detector(feature_detector)
        self.stitcher.set_feature_descriptor(feature_descriptor)
        self.stitcher.set_feature_matcher(feature_matcher)
        self.stitcher.set_homography_method(homography_method)

        # Обработка кадров с камер и получение панорамного изображения
        stitched_image = self.camera_manager.process_frames(self.stitcher)

        # Отображение результата обработки
        if stitched_image is not None:
            self.display_image(stitched_image)

    def display_image(self, image):
        # Конвертация изображения OpenCV в QPixmap для отображения в виджете
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_img = QPixmap.fromImage(QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888))
        
        # Создание виджета для отображения изображения
        label = QLabel(self)
        label.setPixmap(q_img)

        # Добавление виджета в окно
        self.setCentralWidget(label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PanoramaApp()
    window.show()
    sys.exit(app.exec())
