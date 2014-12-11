import numpy
from scipy.spatial import distance

__author__ = 'Emanuele Tamponi'


class Neighborhood(object):

    def __init__(self, inputs, labels, include_center=True):
        self.include_center = include_center
        self.inputs = inputs.copy()
        self.labels = labels.copy()
        self.distance_matrix = distance.squareform(distance.pdist(inputs))

    def iterate(self, radius):
        for row in self.distance_matrix:
            mask = row <= radius
            if not self.include_center:
                mask = numpy.logical_and(mask, row > 0)
            yield (self.inputs[mask], self.labels[mask])
