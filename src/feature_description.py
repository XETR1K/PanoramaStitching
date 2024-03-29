import cv2

def detect_keypoints(image_path, method='SIFT'):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Инициализация детектора ключевых точек
    detectors = {
        'SIFT': cv2.SIFT_create(),
        'SURF': cv2.xfeatures2d.SURF_create(),
        'ORB': cv2.ORB_create()
    }

    detector = detectors[method]

    # Обнаружение ключевых точек и дескрипторов
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)

    return keypoints, descriptors
