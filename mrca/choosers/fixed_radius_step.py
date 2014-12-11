__author__ = 'Emanuele Tamponi'


class FixedRadiusStep(object):

    def __init__(self, finder, smallest_size, profile_dim):
        self.finder = finder
        self.smallest_size = smallest_size
        self.profile_dim = profile_dim

    def choose(self, inputs):
        smallest_size = self._absolute_size(self.smallest_size, inputs)
        smallest_radius = self.finder(smallest_size, inputs)
        radii = [i * smallest_radius for i in range(1, self.profile_dim + 1)]
        return radii

    @staticmethod
    def _absolute_size(size, inputs):
        if isinstance(size, int):
            return size
        else:
            return int(size * (len(inputs) - 1))
