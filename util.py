import glob
import sys

from os import path

import numpy as np

# This allows for 100 angles per location should we need it
SEPARATOR_MULTIPLIER = 100


class DataLoader(object):
    def __init__(self, folder):
        self.coordinates = load_coordinates(path.join(folder, 'coordinates'))
        self.images = load_images(path.join(folder, 'images'))
        self.targets = np.load(
            path.join(folder, 'features_targ'), mmap_mode='r')

        self.distances = haversine(self.coordinates)

        np.fill_diagonal(self.distances, np.inf)
        self.min_dist = self.distances.min(axis=1)
        np.fill_diagonal(self.distances, 0)

    def target2index(self, index):
        return np.searchsorted(self.targets, index, side='right')

    def index2coord(self, index):
        img = self.images[index]
        filename = path.basename(img)
        index = path.splitext(filename)[0].split('_')[0]

        return self.coordinates[int(index)]

    def target2coord(self, index):
        index = self.target2index(index)

        return self.index2coord(index)

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
            data = fd.read().replace(',', '\n').splitlines()
            return np.fromiter(data, dtype=float).reshape(-1, 2)
    else:
        sys.exit('not a file: {}'.format(filename))


def load_images(directory):
    if path.isdir(directory):
        files = glob.glob(path.join(directory, '*.jpg'))
        return sorted(files, key=name_hash)
    else:
        sys.exit('not a directory: {}'.format(directory))


def haversine(x1, x2=None, r=6378137):
    """
    calculates haversine distance of 2 coordinate arrays

    TODO: optimise calculation of square distance matrix?
    we don't need to do N x N calculations...
    """

    x1 = np.deg2rad(x1)
    x2 = x1[:, None] if x2 is None else np.deg2rad(x2)

    dla = x1[..., 0] - x2[..., 0]
    dlo = x1[..., 1] - x2[..., 1]

    a = np.sin(dla * 0.5) ** 2 + np.cos(x1[..., 0]) * \
        np.cos(x2[..., 0]) * np.sin(dlo * 0.5) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return r * c
