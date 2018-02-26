import argparse
import glob
import math

from collections import Counter
from os import path

import cv2
import nmslib
import numpy as np

from scipy.stats import norm

from filters import Particle
from util import DataLoader, haversine


class Search:
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


class VideoProcessor:
    def __init__(self, search, filter, skip=5, web=None):
        self.search = search
        self.filter = filter
        self.skip = skip
        self.web = web

    def process(self, video, truth=None):
        cap = cv2.VideoCapture(video)
        sift = cv2.xfeatures2d.SIFT_create()

        frames = 0
        search = self.search
        coords = search.dataset.coordinates

        filter = self.filter
        filter.reset()

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        votes = np.zeros(len(coords))

        last_seen = None
        visited = []

        if self.web:
            path_fp = open(path.join(self.web, 'coordinates'), 'w+')
            path_gps = open(path.join(self.web, 'gps'), 'w+')

        while cap.isOpened():
            ret, frame = cap.read()
            frames += 1

            if not ret:
                break

            if frames % self.skip != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)

            search.update(des)

            counts = np.bincount(search.coords.ravel())
            votes[:len(counts)] += counts

            if frames % fps == 0:
                most_likely = filter.predict(votes)

                if self.web:
                    tiny = cv2.resize(frame, None, fx=0.5, fy=0.5)
                    cv2.imwrite(path.join(self.web, 'frame.jpg'), tiny)

                    with open(path.join(self.web, 'votes'), 'w+') as fp:
                        for idx, count in enumerate(votes):
                            if count == 0:
                                continue

                            fp.write("{},{},{}\n".format(*coords[idx], count))

                    with open(path.join(self.web, 'particles'), 'w+') as fp:
                        particles = (filter.particles * 0.1).astype(int)
                        for idx, count in enumerate(particles):
                            if count == 0:
                                continue

                            fp.write("{},{},{}\n".format(*coords[idx], count))

                    if most_likely != last_seen:
                        last_seen = most_likely

                        path_fp.write("{},{}\n".format(*coords[most_likely]))
                        path_fp.flush()

                        # calculate MST
                        if most_likely not in visited:
                            visited.append(most_likely)

                            if len(visited) > 1:
                                a = np.array(list(visited))
                                d = search.dataset.distances[a][:, a]
                                mst = a[minimum_spanning_tree(d)]

                                with open(path.join(self.web, 'mst'), 'w+') as fp:
                                    last = None
                                    fmt = "{},{}\n"
                                    for a, b in mst:
                                        if last != a:
                                            fp.write("===================\n")
                                            fp.write(fmt.format(*coords[a]))

                                        fp.write(fmt.format(*coords[b]))
                                        last = b

                coord = coords[most_likely]

                if truth is not None:
                    time = frames // fps

                    idx = truth[:, 0].searchsorted(time) - 1
                    err = haversine(truth[:, 1:][idx], coord)

                    path_gps.write("{},{}\n".format(*truth[:, 1:][idx]))
                    path_gps.flush()

                    print("error: {:.2f}m".format(err))

                votes.fill(0)

        if self.web:
            path_fp.close()
            path_gps.close()


def adjust_gamma(image, gamma=2.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def minimum_spanning_tree(X, copy_X=True):
    """X are edge weights of fully connected graph"""
    if copy_X:
        X = X.copy()

    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []

    # initialize with node 0:
    visited_vertices = [0]
    num_visited = 1
    # exclude self connections:
    np.fill_diagonal(X, np.inf)

    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf
        num_visited += 1
    return np.vstack(spanning_edges)


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

    with open('data/test_videos/2_raw.csv') as fp:
        data = fp.read().replace(',', '\n').splitlines()
        truth = np.fromiter(data, float).reshape(-1, 3)

    pf = Particle(dataset)
    vp = VideoProcessor(search, pf, skip=args.skip, web=args.web)

    vp.process(args.video, truth=truth)


if __name__ == "__main__":
    main()
