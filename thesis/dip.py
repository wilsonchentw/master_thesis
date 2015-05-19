import cv2
import cv2.cv as cv
import numpy as np


def normalize_image(image, norm_size, crop=True):
    if not crop:   
        # Directly resize the image without cropping
        return cv2.resize(image, norm_size)
    else:           
        # Normalize shorter side to norm_size
        height, width, channels = image.shape
        norm_height, norm_width = norm_size
        scale = max(float(norm_height)/height, float(norm_width)/width)
        norm_image = cv2.resize(src=image, dsize=(0, 0), fx=scale, fy=scale)

        # Crop for central part image
        height, width, channels = norm_image.shape
        y, x = (height-norm_height) // 2, (width-norm_width) // 2
        return norm_image[y:y+norm_height, x:x+norm_width]


def get_gradient(image):
    image = image.astype(np.float32)
    sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1) 
    sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y)

    # Truncate angle exceed 2PI
    return magnitude, angle % (np.pi * 2)


def get_clahe(image):
    # Global enhance luminance
    enhance_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = cv2.normalize(enhance_img[:, :, 0], norm_type=cv2.NORM_MINMAX)

    # Perform CLAHE on L-channel
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    luminance = (luminance * 255).astype(np.uint8)
    enhance_img[:, :, 0] = clahe.apply(luminance) / 255.0 * 100.0
    return cv2.cvtColor(enhance_img, cv2.COLOR_LAB2BGR)


def canny_edge(image):
    # Perform Canny edge detection for shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = (gray_image * 255).astype(np.uint8)
    
    mid, std = np.median(gray_image), np.std(gray_image)
    t_lo, t_hi = (mid + std * 0.5), (mid + std * 1.5)
    contour = cv2.Canny(gray_image, t_lo, t_hi, L2gradient=True)
    return contour.astype(np.float32) / 255.0
