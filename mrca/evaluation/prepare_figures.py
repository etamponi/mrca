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
    for probe, classifier in product(evaluation.PROBE_NAMES, evaluation.CLASSIFIER_NAMES):
        synthesis_table(probe, classifier, all_data)


def load_data():
    all_data = {}
    for dataset, cluster_name, n_clusters in product(evaluation.DATASET_NAMES, evaluation.CLUSTER_NAMES,
                                                     evaluation.CLUSTER_NUMS):
        with open("intermediate/{}.int".format(cluster_file_name(dataset, cluster_name, n_clusters))) as f:
            all_data[(dataset, cluster_name, n_clusters)] = cPickle.load(f)
    return all_data


def synthesis_table(probe, classifier, all_data):
    table_data = prepare_synthesis_table_data(probe, classifier, all_data)
    table_name = "synthesis_{}_{}".format(probe, classifier)
    with open("figures/{}.tex".format(table_name), "w") as f:
        f.writelines((r"\begin{table}\centering", NL, r"\label{tab:%s}" % table_name, NL))
        f.writelines((r"\renewcommand{\arraystretch}{1.2}", NL))
        f.writelines((r"\begin{tabularx}{0.9\textwidth}{XXX*{5}{W}|*{5}{W}}", NL))
        f.writelines((r"\toprule", NL))
        f.writelines((
            r" & & & ",
            " & ".join([("\multicolumn{5}{c}{%s}" % evaluation.LEGEND[c]) for c in evaluation.CLUSTER_NAMES]),
            r" \\", NL,
            r"\cmidrule(lr){4-8} \cmidrule(lr){9-13}", NL
        ))
        f.writelines((
            r"\centering $\radius_1$ & \centering $\radius_\profiledim$ & \centering $\profiledim$ & ",
            r" & ".join([(r"\multicolumn{1}{c}{$%d$}" % n_c) for n_c in evaluation.CLUSTER_NUMS]), r" & ",
            r" & ".join([(r"\multicolumn{1}{c}{$%d$}" % n_c) for n_c in evaluation.CLUSTER_NUMS]), r" \\",
            NL
        ))
        for i in range(1, 14):
            f.write(r"\cmidrule(lr){%d-%d}" % (i, i))
        f.write(NL)
        for range_i, table_section in enumerate(table_data):
            f.writelines((
                r"\multirow{5}{*}{%2.0f\%%} & " % (100*evaluation.SIZE_RANGES[range_i][0]),
                r"\multirow{5}{*}{%2.0f\%%} & " % (100*evaluation.SIZE_RANGES[range_i][1])
            ))
            for dim_i, dim_data in enumerate(table_section):
                if dim_i > 0:
                    f.write(r" & & ")
                f.write(r"%d & " % evaluation.PROFILE_DIMS[dim_i])
                for cluster_i, cluster_data in enumerate(dim_data):
                    for n_clusters_i, value in enumerate(cluster_data):
                        value = int(round(value))
                        if value > 50:
                            value = r"\mathbf{%d}" % value
                        f.write("$%s$" % value)
                        if n_clusters_i < 4 or cluster_i < 1:
                            f.write(r"& ")
                f.writelines((r" \\", NL))
            if range_i < 2:
                f.writelines((r"\midrule", NL))
        f.writelines((r"\bottomrule", NL))
        f.writelines((r"\end{tabularx}", NL))
        caption = r"Percentage of datasets on which MRI with {} Probe has correctly estimated {} error rate.".format(
            evaluation.LEGEND[probe],
            evaluation.LEGEND[classifier]
        )
        f.writelines((r"\caption{%s}" % caption, NL))
        f.writelines((r"\end{table}", NL))


def prepare_synthesis_table_data(probe, classifier, all_data):
    n_size_ranges = len(evaluation.SIZE_RANGES)
    n_profile_dims = len(evaluation.PROFILE_DIMS)
    n_cluster_names = len(evaluation.CLUSTER_NAMES)
    n_cluster_nums = len(evaluation.CLUSTER_NUMS)
    table_data = numpy.zeros((n_size_ranges, n_profile_dims, n_cluster_names, n_cluster_nums))
    for range_i in range(n_size_ranges):
        size_range = evaluation.SIZE_RANGES[range_i]
        for dim_i in range(n_profile_dims):
            dim = evaluation.PROFILE_DIMS[dim_i]
            for cluster_i in range(n_cluster_names):
                cluster = evaluation.CLUSTER_NAMES[cluster_i]
                for n_clusters_i in range(n_cluster_nums):
                    n_clusters = evaluation.CLUSTER_NUMS[n_clusters_i]
                    for dataset in evaluation.DATASET_NAMES:
                        mris = all_data[(dataset, cluster, n_clusters)][(probe, dim, size_range)]["mri"]
                        errs = all_data[(dataset, cluster, n_clusters)][(probe, dim, size_range)][classifier]
                        sizes = all_data[(dataset, cluster, n_clusters)][(probe, dim, size_range)]["size"]
                        mris = mris[sizes > 0]
                        errs = errs[sizes > 0]
                        corr, p = pearsonr(mris, errs)
                        if corr > 0 and p <= 0.10:
                            table_data[range_i, dim_i, cluster_i, n_clusters_i] += 1
    table_data = 100 * table_data / len(evaluation.DATASET_NAMES)
    return table_data


if __name__ == '__main__':
    main()
