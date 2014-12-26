import glob
from itertools import product
import os
import re

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from mrca.choosers.linear_radius_step import LinearRadiusStep
from mrca.cluster.manual_centroid_cluster import ManualCentroidCluster
from mrca.probes.imbalance import Imbalance
from mrca.probes.linear_boundary import LinearBoundary


__author__ = 'Emanuele Tamponi'


# CLASSIFIER_NAMES = ["ab", "gb", "ba", "rf", "et"]
CLASSIFIER_NAMES = ["ab", "ba", "rf"]
CLASSIFIERS = {
    "ab": AdaBoostClassifier(n_estimators=100),
    # "gb": GradientBoostingClassifier(n_estimators=100),
    "ba": BaggingClassifier(n_estimators=100),
    "rf": RandomForestClassifier(n_estimators=100),
    # "et": ExtraTreesClassifier(n_estimators=100)
}
N_FOLDS = 5

PROBE_NAMES = ["imb", "lin"]
PROBES = {
    "imb": Imbalance(),
    "lin": LinearBoundary(),
    # "lda": LDAProbe()
}
PROFILE_DIMS = range(5, 26, 5)
SIZE_RANGES = list(product([0.05, 0.10, 0.15, 0.20, 0.25], [0.40, 0.45, 0.50, 0.55, 0.60]))
RADIUS_CHOOSER_CLASS = LinearRadiusStep
RADIUS_FINDER_METHOD = "mean"

CLUSTER_NAMES = ["manual", "kmeans"]
CLUSTER_CLASSES = {
    "manual": ManualCentroidCluster,
    "kmeans": MiniBatchKMeans
}
CLUSTER_NUMS = [2, 3, 4, 5, 6]

LEGEND = {
    "ab": "AdaBoost",
    # "gb": "Grad. Boosting",
    "ba": "Bagging",
    "rf": "Random Forest",
    # "et": "Extr. Rand. Trees",
    "imb": "Imbalance",
    "lin": "Linear Boundary",
    # "lda": "LDA",
    "manual": "Custom Centroids",
    "kmeans": "k-Means"
}


def dataset_names(directory="datasets"):
    dataset_dir = os.path.join(os.path.dirname(__file__), directory)
    names = []
    for dataset_path in glob.glob("{}/*.arff".format(dataset_dir)):
        names.append(re.search(r"([\w\-]+)\.arff", dataset_path).group(1))
    return names


DATASET_NAMES = dataset_names()
# Remove these datasets as we have not enough power to analyze them
DATASET_NAMES.remove("splice")
DATASET_NAMES.remove("letter")


CLUSTERING_CONFS = list(product(DATASET_NAMES, CLUSTER_NAMES, CLUSTER_NUMS))

PROFILE_CONFS = list(product(PROBE_NAMES, PROFILE_DIMS, SIZE_RANGES))
