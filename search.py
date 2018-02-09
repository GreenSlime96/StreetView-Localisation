from collections import Counter

import cv2
import nmslib
import numpy as np

from geopy.distance import distance

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

        # compute mask according to eqn. 1
        mask = d[:, 0] / d[:, -1] <= 0.64

        # prune output, and remove last values from query
        self.queries = queries[mask]
        self.indices = i[mask, :-1]
        self.distances = d[mask, :-1]

    def node_cost(self, i, m):
        return self.distances[i, m]

    def edge_weight(self, i, m, j, n):
        d = self.dataset

        c1 = d.target2coord(self.indices[i, m])
        c2 = d.target2coord(self.indices[j, n])

        return distance(c1, c2).meters

    def search(self):
        pass


def main():
    index = nmslib.init(method='hnsw', space='l2')
    index.loadIndex('data/final.hnsw')

    sift = cv2.xfeatures2d.SIFT_create()
    img = cv2.imread('data/mcdonalds.jpg', cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)

    dataset = DataLoader.create('data')
    search = Search(dataset, index)

    search.update(des)


if __name__ == "__main__":
    main()
