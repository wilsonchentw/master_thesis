import collections
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


def svm_write_problem(filename, label, inst):
    with open(filename, 'w') as fout:
        for y, x in itertools.izip(label, inst):
            output = [str(y)]
            for idx, xi in enumerate(x):
                if abs(xi) > eps:
                    output.append("{0}:{1}".format(idx, xi))
            fout.write(" ".join(output) + "\n")


def preload_list(filename):
    with open(filename, 'r') as fin:
        dataset = collections.defaultdict(list)
        for line in fin:
            path, label = line.strip().split(" ")
            dataset['path'].append(path)
            dataset['label'].append(label)

        return dataset

