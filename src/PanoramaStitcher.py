from features_detection import detect_keypoints
from feature_description import describe_keypoints
from feature_matching import match_keypoints
from homography_estimation import estimate_homography
from image_stitching import warp_images, blend_images

class PanoramaStitcher:
    def __init__(self):
        self.feature_detector = None
        self.feature_descriptor = None
        self.feature_matcher = None
        self.homography_estimator = None

    def set_feature_detector(self, detector):
        #Установка алгоритма обнаружения ключевых точек
        self.feature_detector = detector

    def set_feature_descriptor(self, descriptor):
        #Установка алгоритма описания ключевых точек
        self.feature_descriptor = descriptor

    def set_feature_matcher(self, matcher):
        #Установка алгоритма сопоставления ключевых точек
        self.feature_matcher = matcher

    def set_homography_estimator(self, estimator):
        #Установка алгоритма оценки гомографии
        self.homography_estimator = estimator

    def stitch_images(self, image1, image2):
        #Склеивание двух изображений
        # Обнаружение ключевых точек на первом изображении
        keypoints1 = detect_keypoints(image1, self.feature_detector)
        # Описание ключевых точек на первом изображении
        descriptors1 = describe_keypoints(image1, keypoints1, self.feature_descriptor)

        # Обнаружение ключевых точек на втором изображении
        keypoints2 = detect_keypoints(image2, self.feature_detector)
        # Описание ключевых точек на втором изображении
        descriptors2 = describe_keypoints(image2, keypoints2, self.feature_descriptor)

        # Сопоставление ключевых точек
        matches = match_keypoints(descriptors1, descriptors2, self.feature_matcher)

        # Вычисление гомографии
        homography = estimate_homography(keypoints1, keypoints2, matches, self.homography_estimator)

        # Проективное преобразование и смешивание изображений
        warped_image2 = warp_images(image1, image2, homography)
        stitched_image = blend_images(image1, warped_image2)

        return stitched_image
