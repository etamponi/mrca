from itertools import product
import os
import cPickle

from sklearn import preprocessing

from eole.analysis.dataset_utils import ArffLoader
from mrca import evaluation
from mrca.profile_transform import ProfileTransform


__author__ = 'Emanuele Tamponi'


def main():
    argument_list = product(
        evaluation.DATASET_NAMES, evaluation.PROBE_NAMES, evaluation.PROFILE_DIMS, evaluation.SIZE_RANGES
    )
    evaluation.run_parallel(prepare_dataset_profiles, argument_list)


def prepare_dataset_profiles(dataset, probe_name, profile_dim, size_range):
    smallest_size, largest_size = size_range
    file_name = dataset_profiles_file_name(dataset, probe_name, profile_dim, smallest_size, largest_size)
    if os.path.isfile("intermediate/{}.int".format(file_name)):
        print "{} already done".format(file_name)
        return
    print "{} starting".format(file_name)
    inputs, labels = ArffLoader("datasets/{}.arff".format(dataset)).get_dataset()
    preprocessing.scale(inputs, copy=False)
    transform = ProfileTransform(probe=evaluation.PROBES[probe_name],
                                 chooser=evaluation.RADIUS_CHOOSER_CLASS(
                                     evaluation.RADIUS_FINDER_METHOD, smallest_size, largest_size, profile_dim
                                 ))
    profiles = transform(inputs, labels)
    with open("intermediate/{}.int".format(file_name), "w") as f:
        cPickle.dump(profiles, f)
    print "{} saved".format(file_name)


def dataset_profiles_file_name(dataset, probe_name, profile_dim, smallest_size, largest_size):
    return "{}_profiles_{}_{:02d}_{:02d}_{:02d}".format(
        dataset, probe_name, profile_dim, int(100*smallest_size), int(100*largest_size)
    )


if __name__ == '__main__':
    main()
