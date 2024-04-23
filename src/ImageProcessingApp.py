import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from PanoramaBuilder import PanoramaBuilder

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Panorama Stitching")

        # Список для хранения выбранных изображений
        self.selected_images = []

        # Фрейм для выбора алгоритмов
        self.algorithm_frame = tk.LabelFrame(self.root, text="Выбор алгоритмов")
        self.algorithm_frame.pack(padx=10, pady=10)

        # Обнаружение ключевых точек
        self.feature_detection_label = tk.Label(self.algorithm_frame, text="Обнаружение ключевых точек:")
        self.feature_detection_label.grid(row=0, column=0, sticky="w")
        self.feature_detection_options = ["SIFT", "ORB", "Brisk"]
        self.feature_detection_var = tk.StringVar()
        self.feature_detection_var.set(self.feature_detection_options[0])  # значение по умолчанию
        self.feature_detection_dropdown = tk.OptionMenu(self.algorithm_frame, self.feature_detection_var, *self.feature_detection_options)
        self.feature_detection_dropdown.grid(row=0, column=1, sticky="w")

        # Сопоставление ключевых точек
        self.feature_matching_label = tk.Label(self.algorithm_frame, text="Сопоставление ключевых точек:")
        self.feature_matching_label.grid(row=1, column=0, sticky="w")
        self.feature_matching_options = ["Brute-Force", "K-D-tree"]
        self.feature_matching_var = tk.StringVar()
        self.feature_matching_var.set(self.feature_matching_options[0])  # значение по умолчанию
        self.feature_matching_dropdown = tk.OptionMenu(self.algorithm_frame, self.feature_matching_var, *self.feature_matching_options)
        self.feature_matching_dropdown.grid(row=1, column=1, sticky="w")

        # Оценка гомографии
        self.homography_label = tk.Label(self.algorithm_frame, text="Оценка гомографии:")
        self.homography_label.grid(row=2, column=0, sticky="w")
        self.homography_options = ["RANSAC", "Метод наименьших квадратов"]
        self.homography_var = tk.StringVar()
        self.homography_var.set(self.homography_options[0])  # значение по умолчанию
        self.homography_dropdown = tk.OptionMenu(self.algorithm_frame, self.homography_var, *self.homography_options)
        self.homography_dropdown.grid(row=2, column=1, sticky="w")

        # Фрейм для отображения выбранных изображений с возможностью прокрутки
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Создание горизонтального скроллбара
        self.scrollbar = tk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Создание холста для размещения изображений
        self.image_canvas = tk.Canvas(self.image_frame, xscrollcommand=self.scrollbar.set)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Привязка скроллбара к холсту
        self.scrollbar.config(command=self.image_canvas.xview)

        # Переменная для хранения ссылок на изображения
        self.image_refs = []

        # Чекбокс для отображения выбранных изображений
        self.show_images_var = tk.BooleanVar()
        self.show_images_var.set(True)
        self.show_images_checkbox = tk.Checkbutton(self.root, text="Показать изображения", variable=self.show_images_var, command=self.toggle_images_visibility)
        self.show_images_checkbox.pack()

        # Кнопка для выбора изображений
        self.select_images_button = tk.Button(self.root, text="Выбрать изображения", command=self.select_images)
        self.select_images_button.pack(pady=10)

        # Кнопка для запуска склейки изображений
        self.process_button = tk.Button(self.root, text="Склеить изображения", command=self.process_images)
        self.process_button.pack(pady=10)

    def select_images(self):
        # Функция для выбора и отображения изображений
        self.selected_image_paths = filedialog.askopenfilenames(title="Выберите изображения", filetypes=(("Изображения", "*.png;*.jpg;*.jpeg"),))
        if self.selected_image_paths:
            # Очищаем холст перед добавлением новых изображений
            self.image_canvas.delete("all")
            self.selected_images = []
            self.selected_images = []
            for i, file_path in enumerate(self.selected_image_paths[:4]):  # ограничиваем выбор до 4 изображений
                image = Image.open(file_path)
                image.thumbnail((200, 200))  # уменьшаем изображение для отображения
                photo = ImageTk.PhotoImage(image)
                self.selected_images.append(photo)
                self.image_refs.append(photo)
                self.image_canvas.create_image(20 + i * 220, 20, anchor=tk.NW, image=photo)

            self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))  # обновляем область прокрутки холста
            self.toggle_images_visibility()

    def toggle_images_visibility(self):
        # Функция для переключения видимости выбранных изображений
        state = "normal" if self.show_images_var.get() else "hidden"
        for image in self.image_canvas.find_all():
            self.image_canvas.itemconfig(image, state=state)

    def process_images(self):
        # Функция для запуска склейки изображений
        if len(self.selected_image_paths) < 2:
            return  # Не хватает изображений для склейки

        # Создание экземпляра класса PanoramaBuilder с выбранными алгоритмами
        panorama_builder = PanoramaBuilder(image_paths=self.selected_image_paths, 
                                           feature_detector=self.feature_detection_var.get(),
                                           feature_matcher=self.feature_matching_var.get(),
                                           homography_estimator=self.homography_var.get())
        panorama = panorama_builder.build_panorama()

        # Вывод результата
        plt.imshow(panorama)
        plt.axis('off')
        plt.show()


def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
