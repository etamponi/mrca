from itertools import product
import os
import cPickle
from analysis.dataset_utils import ArffLoader
from mrca import evaluation
from mrca.choosers.linear_size_step import LinearSizeStep
from mrca.profile_transform import ProfileTransform
from mrca.radius_finder import RadiusFinder

__author__ = 'Emanuele Tamponi'


def main():
    for dataset, probe_name, profile_dim, size_range in product(evaluation.DATASET_NAMES, evaluation.PROBE_NAMES,
                                                                evaluation.PROFILE_DIMS, evaluation.SIZE_RANGES):
        prepare_dataset_profiles(dataset, probe_name, profile_dim, size_range)


def prepare_dataset_profiles(dataset, probe_name, profile_dim, size_range):
    smallest_size, largest_size = size_range
    file_name = "{}_profiles_{}_{:02d}_{:02d}_{:02d}".format(dataset, probe_name, profile_dim,
                                                             int(100*smallest_size), int(100*largest_size))
    if os.path.isfile("intermediate/{}.int".format(file_name)):
        print "{} already done"
        return
    print "{} starting".format(file_name)
    inputs, labels = ArffLoader("datasets/{}.arff".format(dataset)).get_dataset()
    transform = ProfileTransform(probe=evaluation.PROBES[probe_name],
                                 chooser=LinearSizeStep(
                                     RadiusFinder("median"), smallest_size, largest_size, profile_dim
                                 ))
    profiles = transform(inputs, labels)
    with open("intermediate/{}.int".format(file_name), "w") as f:
        cPickle.dump(profiles, f)
    print "{} saved".format(file_name)


if __name__ == '__main__':
    main()
