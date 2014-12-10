from __future__ import division

__author__ = 'Emanuele Tamponi'


class Imbalance(object):

    def __init__(self):
        pass

    def __call__(self, x, y, inputs, labels):
        count = (labels == y).sum()
        return 2 * count / len(labels) - 1
