import unittest
import numpy
from mrca.choosers.fixed_step_chooser import FixedStepChooser
from mrca.radius_finder import RadiusFinder

__author__ = 'Emanuele Tamponi'


class TestFixedStepChooser(unittest.TestCase):

    def test_simple_case(self):
        inputs = numpy.asarray([3, 7, 4, 9, 1, 2, 5]).reshape((7, 1))
        expected_radii = [2, 4, 6]
        chooser = FixedStepChooser(RadiusFinder("median"), smallest_size=2, profile_dim=3)
        radii = chooser.choose(inputs)
        self.assertEqual(expected_radii, radii)
