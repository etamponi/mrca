import unittest

import numpy

from mrca.choosers.fixed_radius_step import FixedRadiusStep
from mrca.probes.imbalance import Imbalance
from mrca.profile_transform import ProfileTransform
from mrca.choosers.radius_finder import RadiusFinder


__author__ = 'Emanuele Tamponi'


class TestProfileTransform(unittest.TestCase):

    def test_simple_case(self):
        inputs = numpy.vstack((numpy.random.randn(10, 3) + 2, numpy.random.randn(10, 3) - 1))
        labels = numpy.asarray(list(10*"a") + list(10*"b"))
        transform = ProfileTransform(Imbalance(), FixedRadiusStep(RadiusFinder("mean"), 1, 2))
        profiles = transform(inputs, labels)
        self.assertEqual((20, 2), profiles.shape)
        self.assertTrue(numpy.all(numpy.logical_and(-1 <= profiles, profiles <= 1)))
