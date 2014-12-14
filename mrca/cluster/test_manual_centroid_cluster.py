import unittest

import numpy

from mrca.cluster.manual_centroid_cluster import ManualCentroidCluster
from mrca.mri import MRI


__author__ = 'Emanuele Tamponi'


class TestManualCentroidCluster(unittest.TestCase):

    def test_centroid_sorted_by_mri(self):
        cluster = ManualCentroidCluster(n_clusters=5)
        inputs = numpy.random.randn(50, 20)
        cluster.fit(inputs)
        self.assertEqual((5, 20), cluster.centroids_.shape)
        self.assertTrue(numpy.all(numpy.logical_and(-1 <= cluster.centroids_, cluster.centroids_ <= 1)))
        # Should be sorted in decreasing mri order
        mris = MRI()(cluster.centroids_)
        sorted_mris = numpy.sort(mris)[::-1]
        numpy.testing.assert_array_equal(mris, sorted_mris)

    def test_predict(self):
        cluster = ManualCentroidCluster(n_clusters=2)
        inputs = numpy.zeros((2, 5))
        inputs[0, :] = 1
        cluster.fit(inputs)
        expected_labels = numpy.asarray([1, 0])
        numpy.testing.assert_array_equal(expected_labels, cluster.predict(inputs))
