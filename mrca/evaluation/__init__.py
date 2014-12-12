import glob
from itertools import product
import multiprocessing
import os
import re
import signal
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from mrca.probes.imbalance import Imbalance
from mrca.probes.linear_boundary import LinearBoundary

__author__ = 'Emanuele Tamponi'


CLASSIFIER_NAMES = ["ab", "gb", "ba", "rf", "et"]
CLASSIFIERS = {
    "ab": AdaBoostClassifier(n_estimators=100),
    "gb": GradientBoostingClassifier(n_estimators=100),
    "ba": BaggingClassifier(n_estimators=100),
    "rf": RandomForestClassifier(n_estimators=100),
    "et": ExtraTreesClassifier(n_estimators=100)
}
N_FOLDS = 5

PROBE_NAMES = ["imb", "lin"]
PROBES = {
    "imb": Imbalance(),
    "lin": LinearBoundary()
}
PROFILE_DIMS = range(5, 26, 5)
SIZE_RANGES = [
    (0.05, 0.25),
    (0.10, 0.33),
    (0.15, 0.50)
]

LEGEND = {
    "ab": "AdaBoost",
    "gb": "Grad. Boosting",
    "ba": "Bagging",
    "rf": "Random Forest",
    "et": "Extr. Rand. Trees",
    "imb": "Imbalance",
    "lin": "Linear Boundary"
}

CLUSTER_NUMS = [2, 3, 4, 5, 6]


def dataset_names():
    dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")
    names = []
    for dataset_path in glob.glob("{}/*.arff".format(dataset_dir)):
        names.append(re.search(r"([\w\-]+)\.arff", dataset_path).group(1))
    return names


DATASET_NAMES = dataset_names()


def run_parallel(function, argument_list, processes=4):
    pool = multiprocessing.Pool(processes=processes, initializer=init_worker)
    try:
        for arguments in argument_list:
            pool.apply_async(function, args=tuple(arguments))
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print "Keyboard Interrupt, terminating..."
        pool.terminate()
        pool.join()


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
