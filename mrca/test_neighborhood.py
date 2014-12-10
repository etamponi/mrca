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
        self.assertIsNot(self.labels, neigh.labels)
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
