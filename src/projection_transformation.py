import cv2

def apply_homography(image, homography, target_size):
    # Применение гомографии к изображению
    transformed_image = cv2.warpPerspective(image, homography, target_size)
    return transformed_image
