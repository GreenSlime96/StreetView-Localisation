import argparse
import os
import glob
import random
import sys

from collections import Counter

import cv2
import nmslib
import numpy as np

SEPARATOR_MULTIPLIER = 100

def name_hash(filename):
    filename = os.path.basename(filename)
    split = os.path.splitext(filename)[0].split('_')

    if len(split) != 2:
        sys.exit('unrecognised format: {}'.format(filename))

    return int(split[0]) * SEPARATOR_MULTIPLIER + int(split[1][0])


def build_coordinate_map(filename):
    if os.path.isfile(filename):
        with open(filename) as fd:
            return fd.read().splitlines()
    else:
        sys.exit('not a file: {}'.format(filename))


def load_images(directory):
    if os.path.isdir(directory):
        files = glob.glob(os.path.join(directory, '*.jpg'))
        return sorted(files, key=name_hash)
    else:
        sys.exit('not a directory: {}'.format(directory))


def generate_features(extractor, images, fd):
    descriptors = []
    targets = []

    for i in range(len(images)):
        image = images[i]

        if i % 100 == 0:
            print('Images processed: {}'.format(i))

        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        kp, des = extractor.detectAndCompute(image, None)

        targets.extend([i] * len(des))
        descriptors.extend(des)

    descriptors = np.asarray(descriptors, np.uint8)
    targets = np.asarray(targets, np.uint16)

    np.savez(fd, descriptors=descriptors, targets=targets)


def load_features(extractor, images, filename):
    if not os.path.isfile(filename):
        with open(filename, "wb") as fd:
            generate_features(extractor, images, fd)

    with open(filename, "rb") as fd:
        return np.load(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('coordinates', type=str, help='image-coordinate map')
    parser.add_argument('images', type=str, help='Street View image directory')
    parser.add_argument('features', type=str, help='precomputed SIFT features')
    args = parser.parse_args()

    coordinates = build_coordinate_map(args.coordinates)
    images = load_images(args.images)

    # Perform a length-check to enforce consistency
    if len(images) % len(coordinates) != 0:
        sys.exit('image-coordinate map not consistent')

    sift = cv2.xfeatures2d.SIFT_create()
    data = load_features(sift, images, args.features)

    # try:
    #     data = np.load('data.npz')

    #     print('loaded SIFT features')

    #     descriptors = data['descriptors']
    #     target = data['target']
    # except IOError:

    # print('generating SIFT features')

    # descriptors = []
    # target = []

    # for i in range(len(files)):
    #     fd = files[i]

    #     if i % 100 == 0:
    #         print(i)

    #     image = cv2.imread(fd, cv2.IMREAD_GRAYSCALE)
    #     kp, des = sift.detectAndCompute(image, None)

    #     target.extend([i] * len(des))
    #     descriptors.extend(des)

"""
TODO: prune function
polish pipeline
save features?
"""

# print('generating FLANN tree')

# #     print('saving SIFT features')

# #     np.savez_compressed('data', descriptors=descriptors, target=target)
# # finally:
# #     data.close()

# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)   # or pass empty dictionary

# flann = cv2.FlannBasedMatcher(index_params, search_params)

# # Build random samples
# # indices = random.sample(range(len(descriptors)), 1000)
# # des = np.asarray([descriptors[i] for i in indices])

# # matches = flann.knnMatch(des, descriptors, k=2)

# # for i in range(len(matches)):
# #     match = matches[i]
# #     if match[0].trainIdx != indices[i]:
# #         print('false')

# image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
# kp, des = sift.detectAndCompute(image, None)

# matches = flann.knnMatch(des, descriptors, k=5)
# votes = []

# for m in matches:
#     for k in m:
#         votes.append(target[k.trainIdx])

# c = Counter(votes)

# for i in c.most_common(5):
#     print(files[i])

if __name__ == "__main__":
    main()

