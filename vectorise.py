import argparse
import os

from collections import Counter

import cv2
import numpy as np

from util import load_coordinates, load_images


def build_features(images, filename, bins=16):
    descfile = filename + '_desc'
    targfile = filename + '_targ'
    # histfile = filename + '_hist'

    if os.path.isfile(descfile) or os.path.isfile(targfile):
        sys.exit('filename in use: {}'.format(filename))

    # hist = np.empty(3 * bins * len(images), dtype=np.int32)
    sift = cv2.xfeatures2d.SIFT_create()

    desc = []
    targ = []

    idx = 0
    for i in range(len(images)):
        image = cv2.imread(images[i])
        kp, des = sift.detectAndCompute(image, None)

        # for c in range(3):
        #     hist = cv2.calcHist([image], [c], None, [bins], [0, 256])[:,0]
        #     hists[idx:idx+bins] = hist
        #     idx += bins

        # TODO: update this to use a more compact representation
        targ.extend([i] * len(des))
        desc.extend(des.astype(np.uint8))

        if i % 100 == 0:
            print('Images processed: {}'.format(i))

    # with open(histfile, "wb") as fd:
    #     np.save(fd, hists)

    with open(descfile, "wb") as fd:
        np.save(fd, np.asarray(desc, np.uint8))

    with open(targfile, "wb") as fd:
        np.save(fd, np.asarray(targ, np.uint16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('coordinates', type=str, help='image-coordinate map')
    parser.add_argument('images', type=str, help='Street View image directory')
    parser.add_argument('outfile', type=str, help='computed feature output')
    args = parser.parse_args()

    coordinates = load_coordinates(args.coordinates)
    images = load_images(args.images)

    # Perform a length-check to enforce consistency
    if len(images) % len(coordinates) != 0:
        sys.exit('image-coordinate map not consistent')

    # Use a SIFT feature extractor
    build_features(images, args.outfile)


if __name__ == "__main__":
    main()

