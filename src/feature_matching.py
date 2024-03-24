import cv2

# Сопоставление ключевых точек с использованием выбранного метода
def match_keypoints(descriptors1, descriptors2, method='BruteForce'):
    matcher_methods = {
        'BruteForce': cv2.BFMatcher(cv2.NORM_L2, crossCheck=True),
        'FLANN': cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    }
    matcher = matcher_methods.get(method, None)
    if matcher is None:
        raise ValueError("Unknown method")
    matches = matcher.match(descriptors1, descriptors2)
    return matches
