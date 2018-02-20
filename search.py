import argparse
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

    def coords(self):
        return [self.dataset.coordinates[x] for x in self.coords.flatten()]

    def search(self):
        counts = np.bincount(self.coords.ravel())
        indices = np.argsort(counts)

        return (indices, counts)

    def confidence(self, x_bar=None):
        smoothed = self.smooth()

        n = len(self.dataset.coordinates)
        dist_sq = self.dataset.distances[x_bar][:len(smoothed)] ** 2

        top = np.sum(dist_sq ** 2 * smoothed) / n
        bot = (np.sum(dist_sq) / n) ** 2

        return (top / bot) - 3

    def count(self):
        pass

    def smooth(self):
        if self.smoothed is not None:
            return self.smoothed

        coords = len(self.dataset.coordinates)
        bins = np.bincount(self.coords.ravel(), minlength=coords)

        # TODO: consider reducing the precision to float32 for 2x speedup
        min_distsq = 2 * self.dataset.min_dist ** 2
        distsq = self.dataset.distances ** 2
        wexponent = np.exp(-distsq / min_distsq) * bins[:, None]

        self.smoothed = np.sum(wexponent, axis=0)

        return self.smoothed


class VideoProcessor(object):
    def __init__(self, dataset, states):
        self.dataset = dataset
        self.reset(states)

    def reset(self, states):
        self.votes = []

    def update(self, votes):
        self.votes.append(votes)

    def naive(self):
        votes = self.votes[-1]
        max_votes = np.where(votes == np.amax(votes))[0]

        if len(max_votes) != 1:
            # break tie by finding the vote with most supporters
            # more support = less distance between bits...
            d = self.dataset.distances[max_votes,:][:,max_votes]
            index = max_votes[np.argmin(np.sum(d, axis=0))]
        else:
            # naively pick the most-occuring vote
            index = max_votes[0]

        return index;

    def boon(self):
        votes = self.votes[-1]
        priors = np.ones_like(votes)

        # load priors if they exist
        if len(self.votes) != 1:
            priors = self.votes[-2]





    def PFilter(self):
        pass

    def KFilter(self):
        pass

    def HMM(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str, help='video file')
    parser.add_argument('--web', type=str, default=path.join('web', 'data'),
                        help='web data directory')
    parser.add_argument('--skip', type=int, default=5,
                        help='process every nth frame')
    args = parser.parse_args()

    index = nmslib.init(method='hnsw', space='l2')
    index.loadIndex('data/final.hnsw')

    dataset = DataLoader.create('data')
    search = Search(dataset, index)

    # images = glob.glob('data/test_set/*.jpg')
    cap = cv2.VideoCapture(args.video)
    sift = cv2.xfeatures2d.SIFT_create()

    last_coord = None
    idle_frames = 0
    frames = 0

    votes = np.zeros(len(dataset.coordinates))
    visited = np.zeros(len(dataset.coordinates))
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
    print("FPS: {} \t skip: {}".format(fps, args.skip))

    while cap.isOpened():
        ret, frame = cap.read()
        frames += 1

        if not ret:
            break

        if frames % args.skip != 0 and last_coord:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(gray, None, fx=0.5, fy=0.5)
        kp, des = sift.detectAndCompute(gray, None)

        search.update(des)

        # VIDEO PROCESSING LOGIC

        # update coordinate counts
        if last_coord is not None:
            count = np.bincount(search.coords.ravel())
        else:
            count = search.smooth()

        votes[:len(count)] += count

        if frames % fps == 0:
            top = np.argsort(-votes)

            if not last_coord:
                most_likely = np.argmax(votes)
            else:
                probs = votes / sum(votes)
                top = np.argsort(-votes)

                distances = dataset.distances[last_coord]
                min_dist = max(10, dataset.min_dist[last_coord])

                likelihoods = norm.pdf(
                    distances / ((idle_frames + 2) * min_dist))

                most_likely = np.argmax(likelihoods * votes)

                if most_likely not in top[:10]:
                    # print(votes[top[0]], votes[most_likely],
                    #       dataset.coordinates[top[0]],
                    #       dataset.coordinates[most_likely])
                    # print(search.confidence(top[0]),
                    #       search.confidence(most_likely))

                    most_likely = last_coord
                # else:
                #     # print(search.confidence(most_likely))

            with open(path.join(args.web, 'votes'), 'w') as fp:
                for indices in top:
                    count = votes[indices]
                    coord = dataset.coordinates[indices]

                    fp.write("{},{},{}\n".format(*coord, count))

            tiny = cv2.resize(frame, None, fx=0.5, fy=0.5)
            cv2.imwrite(path.join(args.web, 'frame.jpg'), tiny)

            votes = np.zeros_like(votes)
            frames = 0
            idle_frames += 1

            if most_likely != last_coord:
                idle_frames = 0
                visited += 1

                coord = dataset.coordinates[most_likely]
                print("{},{}".format(*coord), flush=True)

            last_coord = most_likely

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
