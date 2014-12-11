from sympy.external.tests.test_autowrap import numpy

__author__ = 'Emanuele Tamponi'


class FixedSizeStep(object):

    def __init__(self, finder, smallest_size, largest_size, profile_dim):
        self.finder = finder
        self.smallest_size = smallest_size
        self.largest_size = largest_size
        self.profile_dim = profile_dim

    def choose(self, inputs):
        smallest_size = self._absolute_size(self.smallest_size, inputs)
        largest_size = self._absolute_size(self.largest_size, inputs)
        sizes = numpy.linspace(smallest_size, largest_size, self.profile_dim)
        radii = []
        for size in sizes:
            radii.append(self.finder(int(size), inputs))
        return radii

    @staticmethod
    def _absolute_size(size, inputs):
        if isinstance(size, int):
            return size
        else:
            return int(size * (len(inputs) - 1))
