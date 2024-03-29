import cv2

def match_keypoints(descriptors1, descriptors2, method='Brute-Force'):
    # Инициализация объекта сопоставления ключевых точек
    matchers = {
        'Brute-Force': cv2.BFMatcher(),
        'K-D Tree': cv2.FlannBasedMatcher()
    }

    matcher = matchers[method]

    # Сопоставление дескрипторов между изображениями
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Применение порога для отбора лучших сопоставлений
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches
