import cv2

# Описание ключевых точек с использованием выбранного алгоритма
def describe_keypoints(image, keypoints, method='BRIEF'):
    descriptors = None
    descriptor_compute = {
        'BRIEF': cv2.BRIEF_create()
    }
    descriptor = descriptor_compute.get(method, None)
    if descriptor is None:
        raise ValueError("Unknown method")
    keypoints, descriptors = descriptor.compute(image, keypoints)
    return descriptors
