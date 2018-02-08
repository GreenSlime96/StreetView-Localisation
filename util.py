import glob
import sys

from os import path

import numpy as np

# This allows for 100 angles per location should we need it
SEPARATOR_MULTIPLIER = 100

# In [28]: R = 6378137
#     ...: x0 = y0 = None
#     ...: for i in coordinates:
#     ...:     lat, lon = map(lambda x: math.radians(float(x)), i.split(','))
#     ...:     x = R * math.cos(lat) * math.cos(lon)
#     ...:     y = R * math.cos(lat) * math.sin(lon)
#     ...:     if not x0 and not y0:
#     ...:         x0 = x
#     ...:         y0 = y
#     ...:     print("{}\t{}".format(x0 - x, y0 - y))
#     ...:     

class DataLoader(object):
    def __init__(self, folder):
        self.coordinates = load_coordinates(path.join(folder, 'coordinates'))
        self.images = load_images(path.join(folder, 'images'))
        self.targets = np.load(path.join(folder, 'features_targ'), mmap_mode='r')

    @classmethod
    def create(cls, folder='data'):
        if not path.isdir(folder):
            return None

        return cls(folder)


def name_hash(filename):
    filename = path.basename(filename)
    split = path.splitext(filename)[0].split('_')

    if len(split) != 2:
        sys.exit('unrecognised format: {}'.format(filename))

    return int(split[0]) * SEPARATOR_MULTIPLIER + int(split[1])


def load_coordinates(filename):
    if path.isfile(filename):
        with open(filename) as fd:
            return fd.read().splitlines()
    else:
        sys.exit('not a file: {}'.format(filename))


def load_images(directory):
    if path.isdir(directory):
        files = glob.glob(path.join(directory, '*.jpg'))
        return sorted(files, key=name_hash)
    else:
        sys.exit('not a directory: {}'.format(directory))