import itertools

import cv2
import cv2.cv as cv
import numpy as np

eps = 1e-7

def imshow(*images, **kargs):
    # Set how long image will show (in milisecond)
    time = 0 if 'time' not in kargs else kargs['time']
    
    # Modify single channel image to 3-channel by directly tile
    images = [np.atleast_3d(image) for image in images]
    images = [np.tile(image, (1, 1, 3 // image.shape[2])) for image in images]

    # Concatenate image together horizontally
    concat_image = np.hstack(tuple(images))

    # Show image
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", concat_image)
    cv2.waitKey(time) & 0xFF


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


def sliding_window(shape, window, step):
    num_dim = len(tuple(window))
    start = np.zeros(num_dim, np.int32)
    stop = np.array(shape[:num_dim]) - window + 1
    grids = [range(a, b, c) for a, b, c in zip(start, stop, step)]
    for offset in itertools.product(*grids):
        offset = np.array(offset)
        block = [slice(a, b) for a, b in zip(offset, offset+window)]
        yield block


def get_gradient(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y)

    # Truncate angle exceed 2PI
    return magnitude, angle % (np.pi * 2)


def get_clahe(image):
    # Global enhance luminance
    image = image.astype(np.float32)
    enhance_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = cv2.normalize(enhance_img[:, :, 0], norm_type=cv2.NORM_MINMAX)

    # Perform CLAHE on L-channel
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    luminance = (luminance * 255).astype(np.uint8)
    enhance_img[:, :, 0] = clahe.apply(luminance) / 255.0 * 100.0
    enhance_img = cv2.cvtColor(enhance_img, cv2.COLOR_LAB2BGR)

    return enhance_img.astype(float)

