import unittest

import numpy

from mrca.radius_finder import RadiusFinder


__author__ = 'Emanuele Tamponi'


class TestRadiusFinder(unittest.TestCase):

    def test_finder(self):
        inputs = numpy.asarray([
            [1], [5], [2], [8]
        ])
        self.assertAlmostEqual(6, RadiusFinder("max")(2, inputs))
        self.assertAlmostEqual(3.5, RadiusFinder("median")(2, inputs))
        self.assertAlmostEqual(4, RadiusFinder("mean")(2, inputs))
