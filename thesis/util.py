import itertools
import sys

import cv2
import cv2.cv as cv
import numpy as np

eps = 1e-7

class SlidingWindow(object):

    class Block(tuple):
        def __new__(cls, indexes, src, dst):
            self = tuple.__new__(cls, indexes)
            self.src = tuple(src)
            self.dst = tuple(dst)
            return self

    def __init__(self, shape, window, step):
        # Generate grids reference point
        grid = [range(0, e - w + 1, s) for e, w, s in zip(shape, window, step)]
        num_dim = len(grid)

        # Only store effective shape, window, and step
        self.src_shape = tuple(shape[:num_dim])
        self.dst_shape = tuple(len(side) for side in grid)
        self.window = np.array(window[:num_dim])
        self.step = np.array(step[:num_dim])
        self.grid = grid

    def __iter__(self):
        for point in itertools.product(*self.grid):
            indexes = [slice(p, p + w) for p, w in zip(point, self.window)]
            dst_point = np.array(point) // self.step
            yield self.Block(indexes, point, dst_point)


def imshow(*images, **kargs):
    # Set how long image will show (in milisecond)
    time = 0 if 'time' not in kargs else kargs['time']
    
    # Normalize value range if norm flag is set
    if ('norm' in kargs) and kargs['norm']:
        images = [cv2.normalize(image, norm_type=cv2.NORM_MINMAX) 
                  for image in images]

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


"""
def svm_write_problem(filename, label, inst):
    #with open(filename) as fout:
    with sys.stdout as fout:
        for y, x in itertools.izip(label, inst):
            output = [str(y)]
            for idx, xi in enumerate(x):
                if abs(xi) < eps:
                    output.append("{0}:{1}".format(idx, xi))
"""         
