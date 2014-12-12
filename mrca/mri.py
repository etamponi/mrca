import numpy

__author__ = 'Emanuele Tamponi'


class MRI(object):

    def __init__(self):
        pass

    def __call__(self, profiles):
        values = numpy.zeros(len(profiles))
        weights = numpy.asarray([1.0 / (i+1) for i in range(profiles.shape[1])])
        normalization = 1.0 / (2 * weights.sum())
        for i, profile in enumerate(profiles):
            values[i] = normalization * (weights * (1 - profile)).sum()
        return values
