import unittest
import numpy
from mrca.choosers.fixed_size_step import FixedSizeStep
from mrca.radius_finder import RadiusFinder

__author__ = 'Emanuele Tamponi'


class TestFixedSizeStep(unittest.TestCase):

    def test_simple_case(self):
        inputs = numpy.asarray([3, 7, 4, 9, 1, 2, 5]).reshape((7, 1))
        expected_radii = [2, 3, 6]
        chooser = FixedSizeStep(RadiusFinder("median"), smallest_size=2, largest_size=6, profile_dim=3)
        radii = chooser.choose(inputs)
        self.assertEqual(expected_radii, radii)

    def test_float_size(self):
        inputs = numpy.asarray([3, 7, 4, 9, 1, 2, 5]).reshape((7, 1))
        expected_radii = [2, 3, 6]
        chooser = FixedSizeStep(RadiusFinder("median"), smallest_size=0.3334, largest_size=1.0, profile_dim=3)
        radii = chooser.choose(inputs)
        self.assertEqual(expected_radii, radii)
