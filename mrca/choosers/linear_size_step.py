from sympy.external.tests.test_autowrap import numpy

from mrca import choosers
from mrca.choosers.radius_finder import RadiusFinder


__author__ = 'Emanuele Tamponi'


class LinearSizeStep(object):

    def __init__(self, finder_method, smallest_size, largest_size, profile_dim):
        self.finder_method = finder_method
        self.smallest_size = smallest_size
        self.largest_size = largest_size
        self.profile_dim = profile_dim

    def choose(self, inputs):
        finder = RadiusFinder(inputs)
        smallest_size = choosers.absolute_size(self.smallest_size, len(inputs) - 1)
        largest_size = choosers.absolute_size(self.largest_size, len(inputs) - 1)
        sizes = numpy.linspace(smallest_size, largest_size, self.profile_dim)
        radii = []
        for size in sizes:
            radii.append(finder(self.finder_method, int(size)))
        return radii
