import unittest

import numpy

from mrca.mri import MRI


__author__ = 'Emanuele Tamponi'


class TestMRI(unittest.TestCase):

    def test_simple_case(self):
        profiles = numpy.asarray([[0.5, 0.2, 0.1, -0.3]])
        mri = MRI()
        expected_weights = [1.0, 1.0/2, 1.0/3, 1.0/4]
        expected_values = numpy.asarray([
            1.0 / (2*sum(expected_weights)) * sum(w * (1 - p) for w, p in zip(expected_weights, profiles[0]))
        ])
        values = mri(profiles)
        numpy.testing.assert_array_equal(expected_values, values)
