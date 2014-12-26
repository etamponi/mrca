import cPickle
from collections import defaultdict
from math import isnan
from scipy.stats import pearsonr
from mrca.evaluation import *
from mrca.evaluation.prepare_clusters import cluster_file_name

__author__ = 'Emanuele Tamponi'


def main():
    raw_data = prepare_raw_data()

    prepare_data_per_clustering(raw_data)

    prepare_data_per_profile(raw_data)


def prepare_data_per_profile(raw_data):
    file_name = "intermediate/all_data_per_profile.int"
    if os.path.isfile(file_name):
        with open(file_name) as f:
            return cPickle.load(f)
    data = defaultdict(list)
    for clustering_conf, profile_conf, classifier in product(CLUSTERING_CONFS, PROFILE_CONFS, CLASSIFIER_NAMES):
        sizes = raw_data[clustering_conf, profile_conf]["size"]
        mris = raw_data[clustering_conf, profile_conf]["mri"][sizes > 0]
        errs = raw_data[clustering_conf, profile_conf][classifier][sizes > 0]
        corr, _ = pearsonr(mris, errs)
        if isnan(corr):
            if len(mris) == 1:
                corr = 1
            else:
                corr = 0
        data[profile_conf, classifier].append((clustering_conf, corr))
    # Sort each list
    for profile_conf, classifier in product(PROFILE_CONFS, CLASSIFIER_NAMES):
        data[profile_conf, classifier].sort(key=lambda x: -x[1])  # Sort by correlation, descending
    with open(file_name, "w") as f:
        cPickle.dump(data, f)
    return data


def prepare_data_per_clustering(raw_data):
    file_name = "intermediate/all_data_per_clustering.int"
    if os.path.isfile(file_name):
        with open(file_name) as f:
            return cPickle.load(f)
    data = defaultdict(list)
    for clustering_conf, profile_conf, classifier in product(CLUSTERING_CONFS, PROFILE_CONFS, CLASSIFIER_NAMES):
        sizes = raw_data[clustering_conf, profile_conf]["size"]
        mris = raw_data[clustering_conf, profile_conf]["mri"][sizes > 0]
        errs = raw_data[clustering_conf, profile_conf][classifier][sizes > 0]
        corr, _ = pearsonr(mris, errs)
        if isnan(corr):
            if len(mris) == 1:
                corr = 1
            else:
                corr = 0
        data[clustering_conf, classifier].append((profile_conf, corr))
    # Sort each list
    for clustering_conf, classifier in product(CLUSTERING_CONFS, CLASSIFIER_NAMES):
        data[clustering_conf, classifier].sort(key=lambda x: -x[1])  # Sort by correlation, descending
    with open(file_name, "w") as f:
        cPickle.dump(data, f)
    return data


def prepare_raw_data():
    file_name = "intermediate/all_data_raw.int"
    if os.path.isfile(file_name):
        with open(file_name) as f:
            return cPickle.load(f)
    raw_data = {}
    for clustering_conf in CLUSTERING_CONFS:
        with open("intermediate/{}.int".format(cluster_file_name(*clustering_conf))) as f:
            clustering_data = cPickle.load(f)
        for profile_conf in PROFILE_CONFS:
            raw_data[clustering_conf, profile_conf] = clustering_data[profile_conf]
    with open(file_name, "w") as f:
        cPickle.dump(raw_data, f)
    return raw_data


if __name__ == '__main__':
    main()
