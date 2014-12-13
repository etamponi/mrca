from mrca import choosers
from mrca.choosers.radius_finder import RadiusFinder

__author__ = 'Emanuele Tamponi'


class FixedRadiusStep(object):

    def __init__(self, finder_method, smallest_size, profile_dim):
        self.finder_method = finder_method
        self.smallest_size = smallest_size
        self.profile_dim = profile_dim

    def choose(self, inputs):
        finder = RadiusFinder(inputs)
        smallest_size = choosers.absolute_size(self.smallest_size, len(inputs) - 1)
        smallest_radius = finder(self.finder_method, smallest_size)
        radii = [i * smallest_radius for i in range(1, self.profile_dim + 1)]
        return radii
