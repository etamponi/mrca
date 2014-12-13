from __future__ import division

import math

import numpy


__author__ = 'Emanuele Tamponi'


class LinearBoundary(object):

    def __init__(self):
        pass

    def __call__(self, x, y, inputs, labels):
        same_inputs = inputs[labels == y]
        diff_inputs = inputs[labels != y]
        same_n = len(same_inputs)
        diff_n = len(diff_inputs)
        if same_n + diff_n == 0:
            return 0
        if same_n == 0:
            return -1
        if diff_n == 0:
            return +1
        same_mean = same_inputs.mean(axis=0)
        diff_mean = diff_inputs.mean(axis=0)
        if len(inputs) > 2:
            same_sos = self._sum_of_squares(same_inputs - same_mean)
            diff_sos = self._sum_of_squares(diff_inputs - diff_mean)
            cov = 1.0 / (len(inputs) - 2) * (same_sos + diff_sos)
        else:
            # Do not use pooled estimator if we only have two data points
            cov = 1.0 / (len(inputs) - 1) * self._sum_of_squares(inputs - inputs.mean(axis=0))
        try:
            inv_cov = numpy.linalg.inv(cov)
        except numpy.linalg.LinAlgError:
            # If covariance matrix is singular, assume it is a standard random variable
            inv_cov = numpy.eye(inputs.shape[1])
        normal = numpy.dot(inv_cov, same_mean - diff_mean).reshape(inputs.shape[1])
        threshold = 0.5 * numpy.dot(normal, same_mean + diff_mean) + math.log(diff_n / same_n)
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
