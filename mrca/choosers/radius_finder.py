from scipy.spatial import distance

__author__ = 'Emanuele Tamponi'


class RadiusFinder(object):

    def __init__(self, inputs):
        self.distance_matrix = distance.squareform(distance.pdist(inputs))
        self.distance_matrix.sort()

    def __call__(self, method, n_neighbors):
        distances = self.distance_matrix[:, n_neighbors]
        if method == "max":
            return distances.max()
        if method == "median":
            distances.sort()
            mid_point = len(distances) // 2
            return 0.5 * (distances[mid_point] + distances[-1-mid_point])
        if method == "mean":
            return distances.mean()
