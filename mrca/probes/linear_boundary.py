from __future__ import division

import math

import numpy


__author__ = 'Emanuele Tamponi'


class LinearBoundary(object):

    def __init__(self):
        self.inv_cov = None

    def __call__(self, x, y, inputs, labels):
        same_inputs = inputs[labels == y]
        diff_inputs = inputs[labels != y]
        if len(inputs) == 0:
            return 0
        if len(same_inputs) == 0:
            return -1
        if len(diff_inputs) == 0:
            return +1
        same_mean = same_inputs.mean(axis=0)
        diff_mean = diff_inputs.mean(axis=0)
        normal = numpy.dot(self.inv_cov, same_mean - diff_mean).reshape(inputs.shape[1])
        normal_norm = numpy.linalg.norm(normal)
        if normal_norm == 0:
            # If the normal has zero norm, there is no separation between the classes: return undefined complexity
            return 0
        normal /= normal_norm
        threshold = 0.5 * numpy.dot(normal, same_mean + diff_mean)
        return self._normalize(numpy.dot(normal, x) - threshold)

    @staticmethod
    def _sum_of_squares(centered_inputs):
        n_features = centered_inputs.shape[1]
        ret = numpy.zeros((n_features, n_features))
        for x in centered_inputs:
            ret += numpy.dot(x.reshape(n_features, 1), x.reshape(1, n_features))
        return ret

    @staticmethod
    def _normalize(x):
        try:
            return 2 / (1 + math.exp(-x)) - 1
        except OverflowError:
            if x > 0:
                return 1
            else:
                return -1

    def prepare(self, inputs, labels):
        self.inv_cov = None
        if len(inputs) < 2:
            return
        sos = self._sum_of_squares(inputs - inputs.mean())
        cov = sos / (len(inputs) - 1)
        try:
            inv_cov = numpy.linalg.inv(cov)
        except numpy.linalg.LinAlgError:
            # If covariance matrix is singular, assume it is a standard random variable
            inv_cov = numpy.eye(inputs.shape[1])
        self.inv_cov = inv_cov
