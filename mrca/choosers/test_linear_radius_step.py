import unittest

import numpy

from mrca.choosers.linear_radius_step import LinearRadiusStep
from mrca.choosers.radius_finder import RadiusFinder


__author__ = 'Emanuele Tamponi'


class TestLinearRadiusStep(unittest.TestCase):

    def test_simple_case(self):
        inputs = numpy.asarray([3, 7, 4, 9, 1, 2, 5]).reshape((7, 1))
        expected_radii = [2, 3.5, 5]
        chooser = LinearRadiusStep("median", smallest_size=2, largest_size=5, profile_dim=3)
        radii = chooser.choose(inputs)
        self.assertEqual(expected_radii, radii)

    def test_float_size(self):
        inputs = numpy.asarray([3, 7, 4, 9, 1, 2, 5]).reshape((7, 1))
        expected_radii = [2, 3.5, 5]
        chooser = LinearRadiusStep("median", smallest_size=0.3334, largest_size=0.8334, profile_dim=3)
        radii = chooser.choose(inputs)
        self.assertEqual(expected_radii, radii)
