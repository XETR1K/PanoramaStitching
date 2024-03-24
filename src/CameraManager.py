import cv2
from PanoramaStitcher import PanoramaStitcher

class CameraManager:
    def __init__(self, num_cameras):
        # Инициализация объектов камер и экземпляра PanoramaStitcher
        self.num_cameras = num_cameras
        self.cameras = [cv2.VideoCapture(i) for i in range(num_cameras)]
        self.stitcher = PanoramaStitcher()

    def get_frames(self):
        # Получение кадров с камер
        frames = []
        for camera in self.cameras:
            ret, frame = camera.read()
            if ret:
                frames.append(frame)
            else:
                print(f"Не удалось получить кадр с камеры {camera}")
        return frames

    def release_cameras(self):
        # Освобождение ресурсов камер
        for camera in self.cameras:
            camera.release()

    def process_frames(self):
        # Обработка кадров и создание панорамы
        frames = self.get_frames()
        if len(frames) < 2:
            print("Недостаточно камер для создания панорамы.")
            return None
        stitched_image = self.stitcher.stitch_images(frames[0], frames[1])
        for frame in frames[2:]:
            stitched_image = self.stitcher.stitch_images(stitched_image, frame)
        return stitched_image
