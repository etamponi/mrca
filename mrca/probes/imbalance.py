from __future__ import division

__author__ = 'Emanuele Tamponi'


class Imbalance(object):

    def __init__(self):
        pass

    def __call__(self, x, y, inputs, labels):
        if len(labels) == 0:
            return 0
        count = (labels == y).sum()
        return 2 * count / len(labels) - 1

    def prepare(self, inputs, labels):
        pass
