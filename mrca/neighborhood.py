import numpy
from scipy.spatial import distance

__author__ = 'Emanuele Tamponi'


class Neighborhood(object):

    def __init__(self, inputs, labels):
        self.inputs = inputs.copy()
        self.labels = labels.copy()
        self.distance_matrix = distance.squareform(distance.pdist(inputs))

    def iterate(self, radius):
        for row in self.distance_matrix:
            mask = row <= radius
            yield (self.inputs[mask], self.labels[mask])
