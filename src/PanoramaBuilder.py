import cv2
from feature_description import detect_keypoints
from feature_matching import match_keypoints
from homography_estimation import estimate_homography
from projection_transformation import apply_homography
from image_blending import blend_images

class PanoramaBuilder:
    def __init__(self, images, feature_detector='SIFT', feature_matcher='Brute-Force', homography_estimator='RANSAC', image_blender='Simple'):
        self.images = images
        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.homography_estimator = homography_estimator
        self.image_blender = image_blender

    def build_panorama(self):
        # Обнаружение ключевых точек
        keypoints_func = detect_keypoints

        # Сопоставление ключевых точек
        match_keypoints_func = match_keypoints

        # Оценка гомографии
        estimate_homography_func = estimate_homography

        # Преобразование проекций изображений
        apply_homography_func = apply_homography

        # Совмещение изображений
        blend_images_func = blend_images

        keypoints = [keypoints_func(image, self.feature_detector) for image in self.images]

        matches = [match_keypoints_func(keypoints[i][1], keypoints[i+1][1], self.feature_matcher) for i in range(len(keypoints)-1)]

        homographies = [estimate_homography_func(keypoints[i][0], keypoints[i+1][0], self.homography_estimator) for i in range(len(matches))]

        transformed_images = [apply_homography_func(self.images[i+1], homographies[i], self.images[i].shape[:2][::-1]) for i in range(len(homographies))]

        panorama = self.images[0]
        for transformed_image in transformed_images:
            panorama = blend_images_func(panorama, transformed_image, self.image_blender)

        return panorama
