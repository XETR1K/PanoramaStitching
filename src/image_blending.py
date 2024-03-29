import cv2

def blend_images(image1, image2, method='Simple'):
    if method == 'Simple':
        # Простое смешивание изображений
        blended_image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)
    elif method == 'Laplacian':
        # Лапласианное смешивание изображений
        # Преобразование изображений в оттенки серого
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Вычисление пирамид Лапласиана для каждого изображения
        laplacian1 = cv2.Laplacian(gray_image1, cv2.CV_64F)
        laplacian2 = cv2.Laplacian(gray_image2, cv2.CV_64F)

        # Объединение половин изображений
        blended_laplacian = laplacian1 + laplacian2

        # Соединение изображений
        blended_image = image1.copy()
        blended_image[:, int(image1.shape[1]/2):] = blended_laplacian[:, int(image1.shape[1]/2):]

    return blended_image
