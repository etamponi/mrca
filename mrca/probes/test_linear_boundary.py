import unittest
import numpy
from mrca.probes.linear_boundary import LinearBoundary

__author__ = 'Emanuele Tamponi'


class TestLinearBoundary(unittest.TestCase):

    def test_simple_case(self):
        inputs = numpy.vstack((numpy.random.randn(10, 3) + 2, numpy.random.randn(10, 3) - 1))
        labels = numpy.asarray(list(10*"a") + list(10*"b"))
        probe = LinearBoundary()
        for i in range(10):
            x = numpy.random.randn(3)
            y = numpy.random.choice(["a", "b"])
            value = probe(x, y, inputs, labels)
            self.assertTrue(-1 <= value <= 1, "Got {:.3f} on {}-th attempt".format(value, i+1))
        # Assert sign
        self.assertTrue(probe([2, 2, 2], "a", inputs, labels) > 0)
        self.assertTrue(probe([2, 2, 2], "b", inputs, labels) < 0)

    def test_one_class(self):
        inputs = numpy.random.randn(5, 3)
        labels = numpy.asarray(list("aaaaa"))
        probe = LinearBoundary()
        value = probe(numpy.ones(3), "a", inputs, labels)
        self.assertEqual(1, value)
        value = probe(numpy.ones(3), "b", inputs, labels)
        self.assertEqual(-1, value)

    def test_too_few_elements(self):
        inputs = numpy.random.randn(2, 3)
        labels = numpy.asarray(list("ab"))
        probe = LinearBoundary()
        value = probe(numpy.ones(3), "a", inputs, labels)
        self.assertTrue(-1 <= value <= 1)

    def test_zero_covariance(self):
        inputs = numpy.ones((10, 3))
        labels = numpy.asarray(list("aaaaabbbbb"))
        probe = LinearBoundary()
        value = probe(numpy.zeros(3), "a", inputs, labels)
        self.assertTrue(-1 <= value <= 1)

    def test_empty(self):
        inputs = numpy.ones((0, 3))
        labels = numpy.asarray([]).astype('|S1')
        probe = LinearBoundary()
        self.assertEqual(0, probe(numpy.zeros(3), "a", inputs, labels))
