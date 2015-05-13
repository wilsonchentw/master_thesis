import itertools
import math
import sys

import cv2
import cv2.cv as cv
import numpy as np

__all__ = [
    "eps", "imshow", "normalize_image", "sliding_window", 
    "im2row", "row2im", "svm_write_problem", 
]

eps = 1e-7

def imshow(*images, **kargs):
    # Set how long image will show (in milisecond)
    time = 0 if 'time' not in kargs else kargs['time']
    
    # Add single channel image to three channel by directly tile
    images = [np.atleast_3d(image) for image in images]
    images = [np.tile(image, (1, 1, 3 // image.shape[2])) for image in images]
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
        y, x = (height-norm_height)//2, (width-norm_width)//2
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


def im2row(image, window, step):
    (shape, window, step) = map(np.array, (image.shape[:2], window, step))
    num_channel = 1 if len(image.shape) == 2 else image.shape[2]

    num_window = (shape - window) // step + (1, 1)
    if all(num_window > 0):
        num_row = np.prod(num_window)
        dim = np.prod(window) * num_channel
        row = np.empty((num_row, dim), order='C')
        for idx, block in enumerate(sliding_window(shape, window, step)):
            row[idx] = image[block].reshape(-1, order='C')
        return row


def row2im(row, shape, window, step=None):
    if len(row.shape) == 1: 
        return row.reshape(tuple(shape) + (-1,))

    step = window if step is None else step
    num_channel = row.shape[1] // (window[0] * window[1])
    image = np.empty(np.append(shape, num_channel), order='C')
    for idx, block in enumerate(sliding_window(shape, window, step)):
        image[block] = row[idx].reshape(tuple(window) + (num_channel,))
    return image


def svm_write_problem(filename, labels, insts):
    with open(filename) as fout:
        for label, inst in itertools.izip(labels, insts):
            feature = [str(label)]
            for dim, value in enumerate(inst):
                if abs(value) > eps:
                    feature.append("{0}:{1}".format(dim, value))
            fout.write(" ".join(feature) + "\n")
