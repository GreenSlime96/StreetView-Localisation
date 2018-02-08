import argparse
import glob
import os

from collections import Counter

import cv2
import numpy as np

from geopy.distance import distance

from util import load_coordinates, load_images

class Search():
    def __init__(self, coordinates, targets, index, k=5):
        self.coordinates = coordinates
        self.targets = targets
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
        return 
        pass

    def search(self):
        satisfied = False

        while not satisfied:



def hnsw():
    x = np.array(index.knnQueryBatch(des, 6, 1), dtype=(np.int32, np.float32))
    mask = (x[:,1,0] / x[:,1,-1]) <= 0.64
    return x[:,0,:][(x[:,1,0] / x[:,1,-1]) <= 0.64]

def main():
    # index = AnnoyIndex(128, metric='euclidean')
    # index.load('euclidean.annoy')

    sift = cv2.xfeatures2d.SIFT_create()
    img = cv2.imread("data/test.png", cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)

    target = np.load('data/features_targ')

    images = load_images('data/20180130_224839')
    coordinates = load_coordinates('data/pruned')



if __name__ == "__main__":
    main()