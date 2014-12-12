from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from mrca.probes.imbalance import Imbalance
from mrca.probes.linear_boundary import LinearBoundary

__author__ = 'Emanuele Tamponi'


CLASSIFIER_NAMES = ["ab", "gb", "ba", "rf", "et"]
CLASSIFIERS = [
    AdaBoostClassifier(n_estimators=100),
    GradientBoostingClassifier(n_estimators=100),
    BaggingClassifier(n_estimators=100),
    RandomForestClassifier(n_estimators=100),
    ExtraTreesClassifier(n_estimators=100)
]

PROBE_NAMES = ["imb", "lin"]
PROBES = [
    Imbalance(),
    LinearBoundary()
]

PROFILE_SIZES = range(5, 41, 5)

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
