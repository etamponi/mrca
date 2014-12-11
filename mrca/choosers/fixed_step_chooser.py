__author__ = 'Emanuele Tamponi'


class FixedStepChooser(object):

    def __init__(self, finder, smallest_size, profile_dim):
        self.finder = finder
        self.smallest_size = smallest_size
        self.profile_dim = profile_dim

    def choose(self, inputs):
        smallest_radius = self.finder(self.smallest_size, inputs)
        radii = [i * smallest_radius for i in range(1, self.profile_dim + 1)]
        return radii
