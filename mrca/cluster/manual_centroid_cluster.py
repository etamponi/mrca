from __future__ import division

import math

import numpy
from scipy.spatial import distance


__author__ = 'Emanuele Tamponi'


class ManualCentroidCluster(object):

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None

    def fit(self, inputs):
        dim = inputs.shape[1]
        self.cluster_centers_ = numpy.zeros((self.n_clusters, dim))
        for k in range(self.n_clusters):
            for j in range(dim):
                self.cluster_centers_[k][j] = self._centroid_component(k, j, dim)
        return self

    def predict(self, inputs):
        distances = distance.cdist(inputs, self.cluster_centers_)
        return distances.argmin(axis=1)

    def _centroid_component(self, k, j, dim):
        return 2 * self._sigmoid(-10 * (j / (dim - 1) - 1.5 * k/(self.n_clusters - 1) + 0.25)) - 1

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1 + math.exp(-x))
