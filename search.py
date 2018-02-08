import argparse
import glob
import os

from collections import Counter

import cv2
import nmslib
import numpy as np

from geopy.distance import distance

from util import DataLoader

class Search(object):
    def __init__(self, dataset, index, k=5):
        self.dataset = dataset
        self.index = index
        self.k = k

    def update(self, queries):
        # retrieve k+1 results and convert into np arrays
        results = index.knnQueryBatch(queries, k + 1)
        i, d = map(np.array, zip(*results))

        # compute mask according to eqn. 1
        mask = d[:,0] / d[:,-1] <= 0.64

        # prune output, and remove last values
        self.queries = queries[mask]
        self.indices = i[mask,:-1]
        self.distances = d[mask,:-1]

    def node_cost(self, i, m):
        return self.distances[i, m]

    def edge_weight(self, i, m, j, n):
        return distance
        pass

    def search(self):
        satisfied = False



def hnsw():
    x = np.array(index.knnQueryBatch(des, 6, 1), dtype=(np.int32, np.float32))
    mask = (x[:,1,0] / x[:,1,-1]) <= 0.64
    return x[:,0,:][(x[:,1,0] / x[:,1,-1]) <= 0.64]

def main():
    index = nmslib.init(method='hnsw', space='l2')
    index.loadIndex('data/final.hnsw')

    sift = cv2.xfeatures2d.SIFT_create()
    img = cv2.imread('data/test.png', cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)

    dataset = DataLoader.create('data')
    search = Search(dataset, index)

    search.update(des)



if __name__ == "__main__":
    main()