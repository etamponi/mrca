import os
import cPickle

import numpy
import sklearn
from sklearn.cross_validation import StratifiedKFold

from eole.analysis.dataset_utils import ArffLoader
from eole.evaluation import run_parallel
from mrca import evaluation


__author__ = 'Emanuele Tamponi'


def main():
    argument_list = [(dataset, ) for dataset in evaluation.DATASET_NAMES]
    run_parallel(prepare_dataset_predictions, argument_list)


def prepare_dataset_predictions(dataset):
    file_name = "{}_predictions".format(dataset)
    if os.path.isfile("intermediate/{}.int".format(file_name)):
        print "{} already done".format(file_name)
        return
    print "{} starting".format(file_name)
    inputs, labels = ArffLoader("datasets/{}.arff".format(dataset)).get_dataset()
    sklearn.preprocessing.scale(inputs, copy=False)
    predictions = {key: numpy.zeros_like(labels) for key in evaluation.CLASSIFIER_NAMES}
    predictions["oracle"] = labels
    for train_indices, test_indices in StratifiedKFold(labels, n_folds=evaluation.N_FOLDS):
        train_inputs, train_labels = inputs[train_indices], labels[train_indices]
        test_inputs, test_labels = inputs[test_indices], labels[test_indices]
        for name, classifier in evaluation.CLASSIFIERS.iteritems():
            classifier = sklearn.clone(classifier)
            predictions[name][test_indices] = classifier.fit(train_inputs, train_labels).predict(test_inputs)
    with open("intermediate/{}.int".format(file_name), "w") as f:
        cPickle.dump(predictions, f)
    print "{} saved".format(file_name)


if __name__ == '__main__':
    main()
