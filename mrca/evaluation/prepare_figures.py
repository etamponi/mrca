from itertools import product
import cPickle
import numpy
from scipy.stats.stats import pearsonr
from mrca import evaluation
from mrca.evaluation.prepare_clusters import cluster_file_name

__author__ = 'Emanuele Tamponi'


NL = "\n"


def main():
    all_data = load_data()
    for classifier in evaluation.CLASSIFIER_NAMES:
        synthesis_table_per_classifier(classifier, all_data)


def load_data():
    all_data = {}
    for dataset, cluster_name, n_clusters in product(evaluation.DATASET_NAMES, evaluation.CLUSTER_NAMES,
                                                     evaluation.CLUSTER_NUMS):
        with open("intermediate/{}.int".format(cluster_file_name(dataset, cluster_name, n_clusters))) as f:
            all_data[(dataset, cluster_name, n_clusters)] = cPickle.load(f)
    return all_data


def synthesis_table_per_classifier(classifier, all_data):
    table_data = prepare_synthesis_table_per_classifier_data(classifier, all_data)
    table_name = "synthesis_{}".format(classifier)
    with open("figures/{}.tex".format(table_name), "w") as f:
        f.writelines((r"\begin{table}\centering", NL, r"\label{tab:%s}" % table_name, NL))
        f.writelines((r"\renewcommand{\arraystretch}{1.2}", NL))
        f.writelines((r"\begin{tabularx}{1.0\textwidth}{XXX*{10}{W}}", NL))
        f.writelines((r"\toprule", NL))
        f.writelines((
            r" & & & \multicolumn{10}{c}{Number of clusters ($\clusternum$)} \\", NL,
            r"\cmidrule(lr){4-13}", NL
        ))
        f.writelines((
            r"\centering $\radius_1$ & \centering $\radius_\profiledim$ & \centering $\profiledim$ & ",
            r" & ".join([("\multicolumn{2}{c}{$%d$}" % n_c) for n_c in evaluation.CLUSTER_NUMS]), r" \\",
            NL
        ))
        f.writelines((
            r"\cmidrule(lr){1-1} \cmidrule(lr){2-2} \cmidrule(lr){3-3}",
            r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}",
            NL
        ))
        for range_i, table_section in enumerate(table_data):
            f.writelines((
                r"\multirow{10}{*}{%2.0f\%%} & " % (100*evaluation.SIZE_RANGES[range_i][0]),
                r"\multirow{10}{*}{%2.0f\%%} & " % (100*evaluation.SIZE_RANGES[range_i][1])
            ))
            for dim_i, dim_data in enumerate(table_section):
                if dim_i > 0:
                    f.write(r" & & ")
                f.write(r"\multirow{2}{*}{%d} & " % evaluation.PROFILE_DIMS[dim_i])
                for probe_i, probe_data in enumerate(dim_data):
                    if probe_i > 0:
                        f.write(r" & & & ")
                    for n_clusters_i, cluster_data in enumerate(probe_data):
                        value_a = cluster_data[0]
                        value_b = cluster_data[1]
                        if value_a > 17:
                            value_a = r"\mathbf{%d}" % value_a
                        if value_b > 17:
                            value_b = r"\mathbf{%d}" % value_b
                        f.write("$%s$ & $%s$ " % (value_a, value_b))
                        if n_clusters_i < 4:
                            f.write(r"& ")
                    f.writelines((r" \\", NL))
                    if probe_i > 0 and dim_i < 4:
                        f.writelines((r"\cmidrule{3-13}", NL))
            if range_i < 2:
                f.writelines((r"\midrule", NL))
        f.writelines((r"\bottomrule", NL))
        f.writelines((r"\end{tabularx}", NL))
        caption = r"Number of datasets on which MRI has correctly estimated {} error rate.".format(
            evaluation.LEGEND[classifier]
        )
        caption += r" For each number of clusters $\clusternum$ and profile size $\profiledim$, a 2x2 matrix is shown."
        caption += r" On the first row, the Imbalance Probe; on the second row, the Linear Boundary Probe."
        caption += r" On the first column, manual centroid selection; on the second column, k-means centroid selection."
        f.writelines((r"\caption{%s}" % caption, NL))
        f.writelines((r"\end{table}", NL))


def prepare_synthesis_table_per_classifier_data(classifier, all_data):
    n_size_ranges = len(evaluation.SIZE_RANGES)
    n_profile_dims = len(evaluation.PROFILE_DIMS)
    n_probes = len(evaluation.PROBE_NAMES)
    n_cluster_nums = len(evaluation.CLUSTER_NUMS)
    n_cluster_names = len(evaluation.CLUSTER_NAMES)
    table_data = numpy.zeros((n_size_ranges, n_profile_dims, n_probes, n_cluster_nums, n_cluster_names), dtype=int)
    for range_i in range(n_size_ranges):
        size_range = evaluation.SIZE_RANGES[range_i]
        for dim_i in range(n_profile_dims):
            dim = evaluation.PROFILE_DIMS[dim_i]
            for probe_i in range(n_probes):
                probe = evaluation.PROBE_NAMES[probe_i]
                for n_clusters_i in range(n_cluster_nums):
                    n_clusters = evaluation.CLUSTER_NUMS[n_clusters_i]
                    for cluster_i in range(n_cluster_names):
                        cluster = evaluation.CLUSTER_NAMES[cluster_i]
                        for dataset in evaluation.DATASET_NAMES:
                            mris = all_data[(dataset, cluster, n_clusters)][(probe, dim, size_range)]["mri"]
                            errs = all_data[(dataset, cluster, n_clusters)][(probe, dim, size_range)][classifier]
                            sizes = all_data[(dataset, cluster, n_clusters)][(probe, dim, size_range)]["size"]
                            mris = mris[sizes > 0]
                            errs = errs[sizes > 0]
                            corr, p = pearsonr(mris, errs)
                            if corr > 0 and p <= 0.10:
                                table_data[range_i, dim_i, probe_i, n_clusters_i, cluster_i] += 1
    return table_data


if __name__ == '__main__':
    main()
