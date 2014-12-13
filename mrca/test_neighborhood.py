import unittest

import numpy

from mrca.neighborhood import Neighborhood


__author__ = 'Emanuele Tamponi'


class TestNeighborhood(unittest.TestCase):

    def setUp(self):
        self.labels = numpy.asarray(["a", "a", "b"])
        self.inputs = numpy.asarray([5, 6, 7]).reshape(3, 1)

    def test_constructor(self):
        expected_distance_matrix = numpy.asarray([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ])
        neigh = Neighborhood(self.inputs, self.labels)
        self.assertIsNot(self.inputs, neigh.inputs)
        self.assertIsNot(self.labels, neigh.labels)
        numpy.testing.assert_array_equal(self.inputs, neigh.inputs)
        numpy.testing.assert_array_equal(self.labels, neigh.labels)
        numpy.testing.assert_array_equal(expected_distance_matrix, neigh.distance_matrix)

    def test_iterate(self):
        neigh = Neighborhood(self.inputs, self.labels)
        expected_nh = [
            (numpy.asarray([[5], [6]]), numpy.asarray(["a", "a"])),
            (numpy.asarray([[5], [6], [7]]), numpy.asarray(["a", "a", "b"])),
            (numpy.asarray([[6], [7]]), numpy.asarray(["a", "b"]))
        ]
        for i, (inputs, labels) in enumerate(neigh.iterate(radius=1)):
            numpy.testing.assert_array_equal(expected_nh[i][0], inputs)
            numpy.testing.assert_array_equal(expected_nh[i][1], labels)

    def test_include_center(self):
        neigh = Neighborhood(self.inputs, self.labels, include_center=False)
        expected_nh = [
            (numpy.asarray([[6]]), numpy.asarray(["a"])),
            (numpy.asarray([[5], [7]]), numpy.asarray(["a", "b"])),
            (numpy.asarray([[6]]), numpy.asarray(["a"]))
        ]
        for i, (inputs, labels) in enumerate(neigh.iterate(radius=1)):
            numpy.testing.assert_array_equal(expected_nh[i][0], inputs)
            numpy.testing.assert_array_equal(expected_nh[i][1], labels)

    def test_empty(self):
        neigh = Neighborhood(self.inputs, self.labels, include_center=False)
        expected_neigh_inputs = numpy.asarray([]).reshape((0, 1))
        expected_neigh_labels = numpy.asarray([]).astype('|S1')
        for neigh_inputs, neigh_labels in neigh.iterate(radius=0.5):
            numpy.testing.assert_array_equal(expected_neigh_inputs, neigh_inputs,
                                             err_msg="Expected empty, got {}".format(neigh_inputs))
            numpy.testing.assert_array_equal(expected_neigh_labels, neigh_labels)
