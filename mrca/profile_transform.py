import numpy

from mrca.neighborhood import Neighborhood


__author__ = 'Emanuele Tamponi'


class ProfileTransform(object):

    def __init__(self, probe, chooser):
        self.probe = probe
        self.chooser = chooser

    def __call__(self, inputs, labels):
        self.probe.prepare(inputs, labels)
        neigh = Neighborhood(inputs, labels)
        radii = self.chooser.choose(inputs)
        profiles = numpy.zeros((len(inputs), len(radii)))
        for j, radius in enumerate(radii):
            for i, (neigh_inputs, neigh_labels) in enumerate(neigh.iterate(radius)):
                profiles[i][j] = self.probe(inputs[i], labels[i], neigh_inputs, neigh_labels)
        return profiles
