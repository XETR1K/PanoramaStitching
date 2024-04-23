import cv2
import numpy as np

class PanoramaBuilder:
    def __init__(self, image_paths, feature_detector='SIFT', feature_matcher='Brute-Force', homography_estimator='RANSAC'):
        self.image_paths = image_paths
        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.homography_estimator = homography_estimator

    def build_panorama(self):
        if len(self.image_paths) < 2:
            raise ValueError("Not enough images for panorama construction")
        
        images_cv = [cv2.imread(file_path) for file_path in self.image_paths[:2]]
        images_cv_rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images_cv]
        gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images_cv]

        feature_detection_algorithm_dict = {
            "SIFT": cv2.SIFT_create(),
            "ORB": cv2.ORB_create(),
            "Brisk": cv2.BRISK_create()
        }
        
        feature_matching_algorithm_dict = {
            "Brute-Force": cv2.BFMatcher(cv2.NORM_L2),
            "K-D-tree": cv2.FlannBasedMatcher()
        }

        homography_algorithm_dict = {
            "RANSAC": cv2.RANSAC,
            "Метод наименьших квадратов": cv2.LMEDS
        }

        feature_detection_algorithm = feature_detection_algorithm_dict[self.feature_detector]
        keypoints_descriptors = [feature_detection_algorithm.detectAndCompute(image, None) for image in gray_images]

        matcher = feature_matching_algorithm_dict[self.feature_matcher]
        matches = matcher.knnMatch(keypoints_descriptors[0][1], keypoints_descriptors[1][1], k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 4:
            src_pts = np.float32([keypoints_descriptors[0][0][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_descriptors[1][0][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(dst_pts, src_pts, homography_algorithm_dict[self.homography_estimator], 5.0)
        else:
            raise Exception("Not enough matches are found - %d/%d" % (len(good_matches), 4))

        # Получение размеров результирующего изображения
        h1, w1 = images_cv_rgb[0].shape[:2]
        h2, w2 = images_cv_rgb[1].shape[:2]
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        corners1_transformed = cv2.perspectiveTransform(corners1, H)
        corners = np.concatenate((corners2, corners1_transformed), axis=0)
        [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)
        t = [-x_min, -y_min]
        H_translation = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

        # Склеивание изображений
        result = cv2.warpPerspective(images_cv_rgb[1], H_translation.dot(H), (x_max - x_min, y_max - y_min))
        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = images_cv_rgb[0]

        return result
