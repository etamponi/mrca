from scipy.spatial import distance

__author__ = 'Emanuele Tamponi'


class RadiusFinder(object):

    def __init__(self, method):
        self.method = method

    def __call__(self, n_neighbors, inputs):
        distance_matrix = distance.squareform(distance.pdist(inputs))
        distance_matrix.sort()
        distances = distance_matrix[:, n_neighbors]
        if self.method == "max":
            return distances.max()
        if self.method == "median":
            distances.sort()
            mid_point = len(distances) // 2
            return 0.5 * (distances[mid_point] + distances[-1-mid_point])
        if self.method == "mean":
            return distances.mean()