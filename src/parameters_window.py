import tkinter as tk
from tkinter import ttk
from parameters import *
import json

class ParametersWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Настройка параметров")
        self.root.resizable(False, False)  # Запретите изменение размера окна
        
        # Создание фрейма для размещения комбобоксов
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        # Создание комбобоксов для параметров
        self.create_parameter_comboboxes()
        self.load_parameters()

        # Кнопки для сохранения и загрузки параметров
        self.save_button = tk.Button(self.root, text="Сохранить", command=self.save_parameters)
        self.save_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def create_parameter_comboboxes(self):
        # Метод для создания комбобоксов для параметров
        self.parameter_comboboxes = {}

        # Создание комбобоксов для параметра FEATURES_FIND_CHOICES
        label = tk.Label(self.frame, text="Метод поиска ключевых точек")
        label.grid(row=0, column=0, padx=10, pady=5)
        combobox = ttk.Combobox(self.frame, values=list(FEATURES_FIND_CHOICES.keys()), state="readonly")
        combobox.grid(row=0, column=1, padx=10, pady=5)
        combobox.current(0)
        self.parameter_comboboxes['features_find_method'] = combobox

        # Создание комбобоксов для параметра WARP_CHOICES
        label = tk.Label(self.frame, text="Тип искривления")
        label.grid(row=1, column=0, padx=10, pady=5)
        combobox = ttk.Combobox(self.frame, values=WARP_CHOICES, state="readonly")
        combobox.grid(row=1, column=1, padx=10, pady=5)
        combobox.current(0)
        self.parameter_comboboxes['warp_method'] = combobox

        # Создание комбобоксов для параметра SEAM_FIND_CHOICES
        label = tk.Label(self.frame, text="Метод поиска шва")
        label.grid(row=2, column=0, padx=10, pady=5)
        combobox = ttk.Combobox(self.frame, values=list(SEAM_FIND_CHOICES.keys()), state="readonly")
        combobox.grid(row=2, column=1, padx=10, pady=5)
        combobox.current(0)
        self.parameter_comboboxes['seam_find_method'] = combobox

        # Создание комбобоксов для параметра BLEND_CHOICES
        label = tk.Label(self.frame, text="Метод смешивания")
        label.grid(row=3, column=0, padx=10, pady=5)
        combobox = ttk.Combobox(self.frame, values=BLEND_CHOICES, state="readonly")
        combobox.grid(row=3, column=1, padx=10, pady=5)
        combobox.current(0)
        self.parameter_comboboxes['blend_method'] = combobox

        # Создание комбобоксов для параметра EXPOS_COMP_CHOICES
        label = tk.Label(self.frame, text="Метод компенсации экспозиции для коррекции яркости изображений")
        label.grid(row=4, column=0, padx=10, pady=5)
        combobox = ttk.Combobox(self.frame, values=list(EXPOS_COMP_CHOICES.keys()), state="readonly")
        combobox.grid(row=4, column=1, padx=10, pady=5)
        combobox.current(0)
        self.parameter_comboboxes['exposure_comp_method'] = combobox

        # Создание комбобоксов для параметра WAVE_CORRECT_CHOICES
        label = tk.Label(self.frame, text="Метод коррекции волн")
        label.grid(row=5, column=0, padx=10, pady=5)
        combobox = ttk.Combobox(self.frame, values=list(WAVE_CORRECT_CHOICES.keys()), state="readonly")
        combobox.grid(row=5, column=1, padx=10, pady=5)
        combobox.current(0)
        self.parameter_comboboxes['wave_correct_method'] = combobox

        self.parameter_values = {
                    'match_conf': tk.DoubleVar(),  # Параметр match_conf
                    'blend_strength': tk.IntVar(value=3)  # Параметр blend_strength с начальным значением 50
                }

        # Добавляем метки и виджеты Scale для новых параметров
        label = tk.Label(self.frame, text="Пороговое значение для отбора сопоставлений между особенностями изображений")
        label.grid(row=6, column=0, padx=10, pady=5)
        scale = tk.Scale(self.frame, from_=0.3, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.parameter_values['match_conf'])
        scale.grid(row=6, column=1, padx=10, pady=5)

        label = tk.Label(self.frame, text="Сила смешивания при использовании метода блендинга")
        label.grid(row=7, column=0, padx=10, pady=5)
        scale = tk.Scale(self.frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.parameter_values['blend_strength'])
        scale.grid(row=7, column=1, padx=10, pady=5)
    
    def save_parameters(self):
        # Метод для сохранения параметров в JSON файл
        parameters = {}
        for key, combobox in self.parameter_comboboxes.items():
            parameters[key] = combobox.get()
        
        parameters['match_conf'] = self.parameter_values['match_conf'].get()
        parameters['blend_strength'] = self.parameter_values['blend_strength'].get()

        with open('parameters.json', 'w') as f:
            json.dump(parameters, f, indent=4)

    def load_parameters(self):
        # Метод для загрузки параметров из JSON файла
        try:
            with open('parameters.json', 'r') as f:
                parameters = json.load(f)
                for key, value in parameters.items():
                    if key in self.parameter_comboboxes:
                        self.parameter_comboboxes[key].set(value)
                
                if 'match_conf' in parameters:
                    self.parameter_values['match_conf'].set(parameters['match_conf'])
                if 'blend_strength' in parameters:
                    self.parameter_values['blend_strength'].set(parameters['blend_strength'])

        except FileNotFoundError:
            raise ValueError("Файл parameters.json не найден")