import numpy
from mrca.neighborhood import Neighborhood

__author__ = 'Emanuele Tamponi'


class ProfileTransform(object):

    def __init__(self, probe, *radii):
        self.probe = probe
        self.radii = radii

    def __call__(self, inputs, labels):
        neigh = Neighborhood(inputs, labels)
        profiles = numpy.zeros((len(inputs), len(self.radii)))
        for j, radius in enumerate(self.radii):
            for i, (neigh_inputs, neigh_labels) in enumerate(neigh.iterate(radius)):
                profiles[i][j] = self.probe(inputs[i], labels[i], neigh_inputs, neigh_labels)
        return profiles
