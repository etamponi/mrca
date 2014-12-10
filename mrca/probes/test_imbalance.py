from __future__ import division

import unittest
import numpy
from mrca.probes.imbalance import Imbalance

__author__ = 'Emanuele Tamponi'


class TestImbalance(unittest.TestCase):

    def test_simple_case(self):
        inputs = numpy.random.rand(10, 3)
        labels = numpy.asarray(list("aaaaaabbbb"))
        probe = Imbalance()
        expected_value = (4 - 6) / 10
        self.assertAlmostEqual(expected_value, probe([1, 2, 3], "b", inputs, labels))

    def test_empty_case(self):
        probe = Imbalance()
        self.assertEqual(0, probe([1, 2, 3], "a", numpy.asarray([[]]), numpy.asarray([])))
