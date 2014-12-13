import numpy

from mrca import choosers
from mrca.choosers.radius_finder import RadiusFinder


__author__ = 'Emanuele Tamponi'


class LinearRadiusStep(object):

    def __init__(self, finder_method, smallest_size, largest_size, profile_dim):
        self.finder_method = finder_method
        self.smallest_size = smallest_size
        self.largest_size = largest_size
        self.profile_dim = profile_dim

    def choose(self, inputs):
        finder = RadiusFinder(inputs)
        smallest_size = choosers.absolute_size(self.smallest_size, len(inputs) - 1)
        largest_size = choosers.absolute_size(self.largest_size, len(inputs) - 1)
        smallest_radius = finder(self.finder_method, smallest_size)
        largest_radius = finder(self.finder_method, largest_size)
        radii = numpy.linspace(smallest_radius, largest_radius, self.profile_dim).tolist()
        return radii
