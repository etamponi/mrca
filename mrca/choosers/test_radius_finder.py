import unittest

import numpy

from mrca.choosers.radius_finder import RadiusFinder


__author__ = 'Emanuele Tamponi'


class TestRadiusFinder(unittest.TestCase):

    def test_finder(self):
        inputs = numpy.asarray([
            [1], [5], [2], [8]
        ])
        finder = RadiusFinder(inputs)
        self.assertAlmostEqual(6, finder("max", 2))
        self.assertAlmostEqual(3.5, finder("median", 2))
        self.assertAlmostEqual(4, finder("mean", 2))
