from __future__ import division

import math
import numpy

__author__ = 'Emanuele Tamponi'


class ManualCentroidCluster(object):

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centroids_ = None

    def fit(self, inputs):
        dim = inputs.shape[1]
        self.centroids_ = numpy.zeros((self.n_clusters, dim))
        for k in range(self.n_clusters):
            for j in range(dim):
                self.centroids_[k][j] = self._centroid_component(k, j, dim)

    def _centroid_component(self, k, j, dim):
        return 2 * self._sigmoid(-10 * (j / (dim - 1) - 1.5 * k/(self.n_clusters - 1) + 0.25)) - 1

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1 + math.exp(-x))
