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
            raise ValueError("Недостаточно изображений для построения панорамы")
        
        # Загрузка изображений и преобразование в оттенки серого
        images_cv_rgb = [cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB) for file_path in self.image_paths]
        gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images_cv_rgb]

        # Инициализация панорамы первым изображением
        panorama = images_cv_rgb[0]

        # Проход по каждому изображению и склейка их
        for i in range(1, len(images_cv_rgb)):
            # Получение ключевых точек и дескрипторов изображений
            keypoints1, features1 = select_descriptor_methods(gray_images[i], method=self.feature_detector)
            keypoints2, features2 = select_descriptor_methods(panorama, method=self.feature_detector)

            # Сопоставление ключевых точек
            matches = key_points_matching_KNN(features1, features2, ratio=0.7, matcher_method=self.feature_matcher, descriptor_method=self.feature_detector)

            # Оценка гомографии и склейка изображений
            matches, H, status = homography_stitching(keypoints1, keypoints2, matches, method=self.homography_estimator)

            if H is not None:
                h_prev, w_prev = panorama.shape[:2]
                h_curr, w_curr = images_cv_rgb[i].shape[:2]
                corners_prev = np.float32([[0, 0], [0, h_prev], [w_prev, h_prev], [w_prev, 0]]).reshape(-1, 1, 2)
                corners_curr = np.float32([[0, 0], [0, h_curr], [w_curr, h_curr], [w_curr, 0]]).reshape(-1, 1, 2)
                corners_prev_transformed = cv2.perspectiveTransform(corners_prev, H)
                corners = np.concatenate((corners_curr, corners_prev_transformed), axis=0)
                [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
                [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)
                t = [-x_min, -y_min]
                H_translation = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

                result = cv2.warpPerspective(images_cv_rgb[i], H_translation.dot(H), (x_max - x_min, y_max - y_min))
                result[t[1]:h_prev + t[1], t[0]:w_prev + t[0]] = panorama

                panorama = result
            else:
                raise Exception("Оценка гомографии не удалась")

        return panorama

    
def select_descriptor_methods(image, method=None):
    if method == 'SIFT':
        descriptor = cv2.SIFT_create()
    elif method == 'Brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'ORB':
        descriptor = cv2.ORB_create()
        
    (keypoints, features) = descriptor.detectAndCompute(image, None)
    
    return (keypoints, features)

def create_matching_object(matcher_method, descriptor_method, crossCheck):
    bf = None
    if descriptor_method == 'SIFT':
        if matcher_method == 'Brute-Force':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
        elif matcher_method == 'FLANN':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # или передайте пустой словарь
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            return flann
    elif descriptor_method == 'ORB' or descriptor_method == 'Brisk':
        if matcher_method == 'Brute-Force':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
        elif matcher_method == 'FLANN':
            # FLANN с бинарными дескрипторами не имеет смысла, поэтому возвращаем None
            return None
    return bf
    
def key_points_matching_KNN(features1, features2, ratio, matcher_method, descriptor_method):
    matcher = create_matching_object(matcher_method, descriptor_method, crossCheck=False)
    # Вычисление необработанных совпадений и инициализация списка фактических совпадений
    rawMatches = matcher.knnMatch(features1, features2, k=2)
    matches = []

    # Перебор необработанных совпадений
    for m,n in rawMatches:
        # Убедитесь, что расстояние меньше определенного коэффициента друг от друга (Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def homography_stitching(keypoints1, keypoints2, matches, method='RANSAC', reprojThresh=3.0):
    keypoints1 = np.float32([keypoint.pt for keypoint in keypoints1])
    keypoints2 = np.float32([keypoint.pt for keypoint in keypoints2])

    if len(matches) > 4:
        points1 = np.float32([keypoints1[m.queryIdx] for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx] for m in matches])

        H, status = estimate_homography_method(points1, points2, method=method, ransacReprojThreshold=reprojThresh)

        return matches, H, status
    else:
        return None

def estimate_homography_method(points1, points2, method='RANSAC', ransacReprojThreshold=3.0, confidence=0.99):
    if method == 'RANSAC':
        return cv2.findHomography(points1, points2, cv2.RANSAC, ransacReprojThreshold)
    elif method == 'LMEDS':
        return cv2.findHomography(points1, points2, cv2.LMEDS)
