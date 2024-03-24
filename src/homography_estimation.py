import cv2
import numpy as np

# Оценка гомографии с использованием выбранного метода
def estimate_homography(keypoints1, keypoints2, matches, method='RANSAC'):
    methods = {
        'RANSAC': cv2.RANSAC,
    }
    method_value = methods.get(method, None)
    if method_value is None:
        raise ValueError("Unknown method")
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, method_value, 5.0)
    return homography
