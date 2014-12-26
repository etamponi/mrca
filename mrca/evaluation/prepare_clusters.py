import cPickle
from itertools import product

import numpy
from sklearn.metrics.metrics import zero_one_loss
from eole.evaluation import run_parallel

from mrca import evaluation
from mrca.evaluation.prepare_profiles import dataset_profiles_file_name
from mrca.mri import MRI


__author__ = 'Emanuele Tamponi'


def main():
    arg_list = product(evaluation.DATASET_NAMES, evaluation.CLUSTER_NAMES, evaluation.CLUSTER_NUMS)
    run_parallel(prepare_dataset_clusters, arg_list)


def prepare_dataset_clusters(dataset, cluster_name, n_clusters):
    file_name = cluster_file_name(dataset, cluster_name, n_clusters)
    # if os.path.isfile("intermediate/{}.int".format(file_name)):
    #     print "{} already done".format(file_name)
    #     return
    print "{} starting".format(file_name)
    with open("intermediate/{}_predictions.int".format(dataset)) as f:
        predictions = cPickle.load(f)
    results = {}
    for probe_name, profile_dim, size_range in product(evaluation.PROBE_NAMES, evaluation.PROFILE_DIMS,
                                                       evaluation.SIZE_RANGES):
        smallest_size, largest_size = size_range
        profiles_file_name = dataset_profiles_file_name(dataset, probe_name, profile_dim, smallest_size, largest_size)
        with open("intermediate/{}.int".format(profiles_file_name)) as f:
            profiles = cPickle.load(f)
        results[(probe_name, profile_dim, size_range)] = prepare_cluster(
            profiles, predictions, cluster_name, n_clusters
        )
    with open("intermediate/{}.int".format(file_name), "w") as f:
        cPickle.dump(results, f)
    print "{} saved".format(file_name)


def prepare_cluster(profiles, predictions, cluster_name, n_clusters):
    true_labels = predictions["oracle"]
    cluster_class = evaluation.CLUSTER_CLASSES[cluster_name]
    cluster = cluster_class(n_clusters)
    clus_labels = cluster.fit(profiles).predict(profiles)
    results = {
        "mri": cluster_mri(profiles, clus_labels, n_clusters),
        "size": cluster_size(clus_labels, n_clusters),
        "centroid": cluster.cluster_centers_
    }
    sorting_indices = results["mri"].argsort()
    results["mri"] = results["mri"][sorting_indices]
    results["size"] = results["size"][sorting_indices]
    results["centroid"] = results["centroid"][sorting_indices]
    for classifier in evaluation.CLASSIFIER_NAMES:
        pred_labels = predictions[classifier]
        results[classifier] = cluster_error(true_labels, pred_labels, clus_labels, n_clusters)[sorting_indices]
    return results


def cluster_size(clus_labels, n_clusters):
    return numpy.asarray([(clus_labels == c).sum() for c in range(n_clusters)])


def cluster_mri(profiles, clus_labels, n_clusters):
    mri = MRI()
    ret = numpy.zeros(n_clusters)
    for c in range(n_clusters):
        ret[c] = mri(profiles[clus_labels == c]).mean()
    return ret


def cluster_error(true_labels, pred_labels, clus_labels, n_clusters):
    ret = numpy.zeros(n_clusters)
    for c in range(n_clusters):
        ret[c] = zero_one_loss(true_labels[clus_labels == c], pred_labels[clus_labels == c])
    return ret


def cluster_file_name(dataset, cluster_name, n_clusters):
    return "{}_{}_cluster_{}".format(dataset, cluster_name, n_clusters)


if __name__ == '__main__':
    main()
