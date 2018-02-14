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
        counts = np.bincount(self.coords.ravel()).astype(np.uint16)
        indices = np.argsort(counts)

        return (indices, counts)

    def confidence(self):
        smoothed = self.smooth()

        expected = np.argmax(smoothed)
        min_dist = self.dataset.min_dist[expected] ** 4

        # TODO: need to figure out how this actually works
        confidence = 0
        for i, c in enumerate(smoothed):
            confidence += self.dataset.distances[expected][i] ** 4 * c

        return -3 + confidence / min_dist

    def count(self):
        pass

    def smooth(self):
        if self.smoothed is not None:
            return self.smoothed

        bins = np.bincount(self.coords.ravel())

        # TODO: consider reducing the precision to float32 for 2x speedup
        min_distsq = 2 * self.dataset.min_dist[:len(bins)] ** 2
        distsq = self.dataset.distances[:len(bins), :len(bins)] ** 2
        wexponent = np.exp(-distsq / min_distsq) * bins[:, None]

        self.smoothed = np.sum(wexponent, axis=0)

        return self.smoothed


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
    cap = cv2.VideoCapture('data/test_videos/1.mp4')
    sift = cv2.xfeatures2d.SIFT_create()

    last_coord = None
    idle_frames = 0
    frames = 0

    votes = np.zeros(len(dataset.coordinates))
    visited = np.zeros(len(dataset.coordinates))
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
    print("Input FPS: {}".format(fps))

    while cap.isOpened():
        ret, frame = cap.read()
        frames += 1

        if not ret:
            break

        if frames % 5 != 0 and last_coord:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        search.update(des)

        # VIDEO PROCESSING LOGIC

        # update coordinate counts
        if not last_coord:
            count = search.smooth()
        else:
            count = np.bincount(search.coords.ravel())

        votes[:len(count)] += count

        if frames % fps == 0:

            if not last_coord:
                most_likely = np.argmax(votes)
            else:
                top_20 = np.argsort(-votes)[:50]
                n_votes = votes[top_20]

                distances = dataset.distances[last_coord][top_20]
                min_dist = max(10, dataset.min_dist[last_coord])

                # print("// {:.2f}".format(min_dist))

                likelihoods = norm.pdf(
                    distances / ((idle_frames + 2) * min_dist))

                most_likely = top_20[np.argmax(likelihoods * n_votes)]

                # if distances[most_likely] > 100:
                #     most_likely = last_coord
                # else:
                #     most_likely = top_20[most_likely]

                # print("// ", end="")
                # print(votes[most_likely], np.amax(votes),
                #       likelihoods[most_likely], likelihoods[np.argmax(votes)],
                #       dataset.coordinates[np.argmax(votes)])

            votes = np.zeros_like(votes)
            frames = 0
            visited += 1
            idle_frames += 1

            if most_likely != last_coord:
                idle_frames = 0
                coord = dataset.coordinates[most_likely]
                print("{{lat:{},lng:{}}},".format(coord[0], coord[1]))

            last_coord = most_likely

            # maxprob = 0
            # maxcoord = None
            # maxdist = None

            # if lastcoord and velocity:
            #     newcoord = (lastcoord[0] + velocity[0],
            #                 lastcoord[1] + velocity[1])

            #     # d_v = distance(newcoord, lastcoord).meters
            #     # d_v = d_v if d_v else 12

            #     for coord, count in votes:
            #         dist = distance(newcoord, c).meters

            #         prob = norm.pdf(dist / 50) * n

            #         if prob > maxprob:
            #             maxdist = dist
            #             maxprob = prob
            #             maxcoord = c

            #     # if maxdist > 100:
            #     #     maxcoord = lastcoord

            # else:
            #     # topmost item
            #     maxcoord = common[0][0]
            #     lastcoord = maxcoord

            # coord = maxcoord

            # if coord != lastcoord or not velocity:
            #     velocity = (coord[0] - lastcoord[0], coord[1] - lastcoord[1])
            #     print("{{lat:{},lng:{}}},".format(coord[0], coord[1]))

            # lastcoord = coord

            # coord = None

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
