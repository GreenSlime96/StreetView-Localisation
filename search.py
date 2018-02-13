import glob
import math

from collections import Counter
from os import path

import cv2
import nmslib
import numpy as np

from geopy.distance import distance
from scipy.stats import norm

from util import DataLoader

# In [27]: %timeit np.bincount(vf(search.indices.flatten())).argsort()[-5:][::-1]
# 3.69 ms ± 28.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# In [28]: %timeit Counter([funky(x) for x in search.indices.flatten()]).most_common(5)
# 3.83 ms ± 9.16 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# In [29]: %timeit Counter([funky(x) for x in search.indices.flatten()]).most_common(5)
# 3.81 ms ± 9.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# In [30]: %timeit np.bincount(vf(search.indices.flatten())).argsort()[-5:][::-1]
# 3.84 ms ± 172 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


class Search(object):
    def __init__(self, dataset, index, k=5):
        self.dataset = dataset
        self.index = index
        self.k = k

    def update(self, queries):
        # retrieve k+1 results and convert into np arrays
        results = self.index.knnQueryBatch(queries, self.k + 1)
        i, d = map(np.array, zip(*results))

        # compute pruning mask to filter features
        mask = d[:, 0] / d[:, -1] <= 0.64

        # prune output, and remove last values from query
        self.queries = queries[mask]
        self.indices = i[mask, :-1]
        self.distances = d[mask, :-1]

        # store these values which are frequently accessed
        self.imgidx = self.dataset.target2index(self.indices)
        self.coords = (self.imgidx / 5).astype(np.uint32)

        # reset for re-memoisation
        self.smoothed = None

    def node_cost(self, i, m):
        return self.distances[i, m]

    def edge_weight(self, i, m, j, n):
        d = self.dataset

        c1 = d.target2coord(self.indices[i, m])
        c2 = d.target2coord(self.indices[j, n])

        return distance(c1, c2).meters

    def coords(self):
        return [self.dataset.coordinates[x] for x in self.coords.flatten()]

    def search(self):
        counts = np.bincount(self.coords.flatten()).astype(np.uint16)
        indices = np.argmax(counts)

        return (indices, counts)

    def confidence(self):
        smoothed = self.smooth()

        expected = np.argmax(smoothed)
        mindist = self.dataset.mindist[expected] ** 4

        # TODO: need to figure out how this actually works
        confidence = 0
        for i, c in enumerate(smoothed):
            confidence += self.dataset.distances[expected][i] ** 4 * c

        return -3 + confidence / mindist

    def count(self):
        pass

    def smooth(self):
        bins = np.bincount(self.coords.flatten())

        mindistsq = 2 * self.dataset.mindist[:len(bins)] ** 2
        distsq = self.dataset.distances[:len(bins),:len(bins)] ** 2

        return np.sum(np.exp(-distsq/mindistsq) * bins[:,None], axis=0)



class VideoProcessor(object):
    def __init__(self, fps, skip=6):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.started = False
        self.fps = fps

    def process(self, frame):
        pass

    def finalise(self):
        pass


def main():
    index = nmslib.init(method='hnsw', space='l2')
    index.loadIndex('data/final.hnsw')

    dataset = DataLoader.create('data')
    search = Search(dataset, index)

    # images = glob.glob('data/test_set/*.jpg')
    cap = cv2.VideoCapture('data/test_videos/6.mp4')
    sift = cv2.xfeatures2d.SIFT_create()

    lastcoord = None
    velocity = None

    frames = 0
    coords = []
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
    print("Input FPS: {}".format(fps))

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        search.update(des)

        coords.extend([dataset.target2coord(x)
                       for x in search.indices.flatten()])

        if frames == fps:
            frames = 0
            coord = None

            # compute probability of top-10 votes
            common = Counter(coords).most_common(50)
            coords = []

            # total = sum([n for c, n in common])
            # common = [(c, n / total) for c, n in common]

            maxprob = 0
            maxcoord = None
            maxdist = None

            if lastcoord and velocity:
                newcoord = (lastcoord[0] + velocity[0],
                            lastcoord[1] + velocity[1])

                # d_v = distance(newcoord, lastcoord).meters
                # d_v = d_v if d_v else 12

                for c, n in common:
                    dist = distance(newcoord, c).meters

                    prob = norm.pdf(dist / 50) * n

                    if prob > maxprob:
                        maxdist = dist
                        maxprob = prob
                        maxcoord = c

                # if maxdist > 100:
                #     maxcoord = lastcoord

            else:
                # topmost item
                maxcoord = common[0][0]
                lastcoord = maxcoord

            coord = maxcoord

            if coord != lastcoord or not velocity:
                velocity = (coord[0] - lastcoord[0], coord[1] - lastcoord[1])
                print("{{lat:{},lng:{}}},".format(coord[0], coord[1]))

            lastcoord = coord

        frames += 1

    cap.release()

    # for image in images:
    #     print("working on: {}".format(image))

    #     img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #     kp, des = sift.detectAndCompute(img, None)

    #     search.update(des)

    #     filename = path.splitext(image)[0]

    # with open(filename + '_base', 'w') as fp:
    #     coords = [dataset.target2coord(x)
    #               for x in search.i[:, :-1].flatten()]
    #     for c in coords:
    #         fp.write(
    #             "new google.maps.LatLng({}, {}),\n".format(c[0], c[1]))

    # with open(filename + '_prune', 'w') as fp:
    #     coords = [dataset.target2coord(x)
    #               for x in search.indices[:, :-1].flatten()]
    #     for c in coords:
    #         fp.write(
    #             "new google.maps.LatLng({}, {}),\n".format(c[0], c[1]))

    # with open(filename + '_smooth', 'w') as fp:
    #     coords = []
    #     for i, c in enumerate(search.smooth()):
    #         coords.extend([dataset.coordinates[i]] * int(c))

    #     print("{}: ".format(search.confidence()))

    #     for c in coords:
    #         fp.write(
    #             "new google.maps.LatLng({}, {}),\n".format(c[0], c[1]))


if __name__ == "__main__":
    main()
