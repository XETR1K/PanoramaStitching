import cv2

# Проективное преобразование изображений
def warp_images(image1, image2, homography):
    height, width = image1.shape[:2]
    warped_image2 = cv2.warpPerspective(image2, homography, (width, height))
    return warped_image2

# Смешивание перекрывающихся областей изображений
def blend_images(image1, warped_image2):
    mask = cv2.cvtColor(warped_image2, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 255
    mask_inv = cv2.bitwise_not(mask)
    image1_masked = cv2.bitwise_and(image1, image1, mask=mask_inv)
    warped_image2_masked = cv2.bitwise_and(warped_image2, warped_image2, mask=mask)
    blended_image = cv2.add(image1_masked, warped_image2_masked)
    return blended_image
