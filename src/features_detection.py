import cv2

# Обнаружение ключевых точек с использованием выбранного алгоритма
def detect_keypoints(image, method='SIFT'):
    detectors = {
        'SIFT': cv2.SIFT_create(),
        'SURF': cv2.xfeatures2d.SURF_create(),
        'ORB': cv2.ORB_create()
    }
    detector = detectors.get(method, None)
    if detector is None:
        raise ValueError("Unknown method")
    keypoints = detector.detect(image, None)
    return keypoints

