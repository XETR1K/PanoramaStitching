import cv2

def estimate_homography(keypoints1, keypoints2, method='RANSAC'):
    # Извлечение координат ключевых точек
    points1 = [keypoint.pt for keypoint in keypoints1]
    points2 = [keypoint.pt for keypoint in keypoints2]

    # Оценка гомографии
    methods = {
        'DLT': 0,
        'RANSAC': cv2.RANSAC,
        'LSQ': 0
    }

    method_value = methods[method]

    homography, _ = cv2.findHomography(points1, points2, method_value)

    return homography
