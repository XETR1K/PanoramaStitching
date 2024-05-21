import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from parameters_window import ParametersWindow
import matplotlib.pyplot as plt
from ImageStitcher import ImageStitcher
import cv2
import json

class StitchingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stitching App")
        self.root.resizable(False, False)
        self.selected_image_paths = []
        self.selected_images = []

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
        self.process_button = tk.Button(self.root, text="Склеить изображения", command=self.stitch)
        self.process_button.pack(pady=10)

        # Кнопка для открытия окна настройки параметров
        self.parameters_button = tk.Button(self.root, text="Настройка параметров", command=self.open_parameters_window)
        self.parameters_button.pack(pady=10)

    def select_images(self):
        # Функция для выбора и отображения изображений
        self.selected_image_paths = filedialog.askopenfilenames(title="Выберите изображения", filetypes=(("Изображения", "*.png;*.jpg;*.jpeg"),))
        if self.selected_image_paths:
            # Очищаем холст перед добавлением новых изображений
            self.image_canvas.delete("all")
            self.selected_images = []
            self.image_refs = []
            for i, file_path in enumerate(self.selected_image_paths):
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

    def load_parameters_from_json(self, filename):
        # Метод для загрузки параметров из JSON файла
        try:
            with open(filename, 'r') as f:
                parameters = json.load(f)
                return parameters
        except FileNotFoundError:
            print("Файл parameters.json не найден")
            return {}

    def stitch(self):
        if not self.selected_image_paths:
            tk.messagebox.showwarning("Нет выбранных изображений", "Пожалуйста, выберите изображения для сшивания.")
            return

        stitcher = ImageStitcher()
        parameters = self.load_parameters_from_json('parameters.json')

        stitcher.set_features(parameters.get('features_find_method', 'default_value'))
        stitcher.set_blend(parameters.get('blend_method', 'default_value'))
        stitcher.set_warp(parameters.get('warp_method', 'default_value'))
        stitcher.set_seam(parameters.get('seam_find_method', 'default_value'))
        stitcher.set_expos_comp(parameters.get('exposure_comp_method', 'default_value'))
        stitcher.set_wave_correct(parameters.get('wave_correct_method', 'default_value'))

        try:
            stitcher.load_images(self.selected_image_paths)
            result_image = stitcher.stitch_images()
            # Вывод результата
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        except Exception as e:
            tk.messagebox.showerror("Ошибка", f"Произошла ошибка при сшивании изображений: {e}")

    def open_parameters_window(self):
        # Функция для открытия окна настройки параметров
        parameters_window = tk.Toplevel(self.root)
        parameters_window.grab_set()  # Захватываем все события в этом окне
        ParametersWindow(parameters_window)

if __name__ == "__main__":
    root = tk.Tk()
    app = StitchingApp(root)
    root.mainloop()
