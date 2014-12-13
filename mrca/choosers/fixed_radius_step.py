from mrca import choosers

__author__ = 'Emanuele Tamponi'


class FixedRadiusStep(object):

    def __init__(self, finder, smallest_size, profile_dim):
        self.finder = finder
        self.smallest_size = smallest_size
        self.profile_dim = profile_dim

    def choose(self, inputs):
        smallest_size = choosers.absolute_size(self.smallest_size, len(inputs) - 1)
        smallest_radius = self.finder(smallest_size, inputs)
        radii = [i * smallest_radius for i in range(1, self.profile_dim + 1)]
        return radii
