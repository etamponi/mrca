from sympy.external.tests.test_autowrap import numpy

__author__ = 'Emanuele Tamponi'


class FixedSizeStep(object):

    def __init__(self, finder, smallest_size, largest_size, profile_dim):
        self.finder = finder
        self.smallest_size = smallest_size
        self.largest_size = largest_size
        self.profile_dim = profile_dim

    def choose(self, inputs):
        sizes = numpy.linspace(self.smallest_size, self.largest_size, self.profile_dim)
        radii = []
        for size in sizes:
            radii.append(self.finder(int(size), inputs))
        return radii
