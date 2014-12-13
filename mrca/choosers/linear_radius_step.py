import numpy
from mrca import choosers

__author__ = 'Emanuele Tamponi'


class LinearRadiusStep(object):

    def __init__(self, finder, smallest_size, largest_size, profile_dim):
        self.finder = finder
        self.smallest_size = smallest_size
        self.largest_size = largest_size
        self.profile_dim = profile_dim

    def choose(self, inputs):
        smallest_size = choosers.absolute_size(self.smallest_size, len(inputs) - 1)
        largest_size = choosers.absolute_size(self.largest_size, len(inputs) - 1)
        smallest_radius = self.finder(smallest_size, inputs)
        largest_radius = self.finder(largest_size, inputs)
        radii = numpy.linspace(smallest_radius, largest_radius, self.profile_dim).tolist()
        return radii
