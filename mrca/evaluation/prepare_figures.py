from itertools import product, cycle
import cPickle
import matplotlib
import numpy
from scipy.stats.stats import pearsonr
from mrca.evaluation import *
from mrca.evaluation.prepare_clusters import cluster_file_name

__author__ = 'Emanuele Tamponi'


NL = "\n"


def main():
    # configure_matplotlib()
    from matplotlib import pyplot

    all_data = load_data()

    # for probe, classifier in product(PROBE_NAMES, CLASSIFIER_NAMES):
    #     synthesis_table(probe, classifier, all_data)

    size_ranges = [(0.05, 0.25)]  # Only one size range
    cluster_names = ["manual"]    # Show only manual centroid clustering
    classifier_names = ["rf"]     # Plot only against RF
    # for params in product(DATASET_NAMES, cluster_names, PROBE_NAMES, PROFILE_DIMS, size_ranges, classifier_names):
    #     draw_plot(pyplot, all_data, *params)
    cluster_nums = [3, 4, 5]
    for params in product(cluster_nums, cluster_names, PROBE_NAMES, PROFILE_DIMS, SIZE_RANGES, classifier_names):
        draw_correlation_with_size(pyplot, all_data, *params)


def draw_correlation_with_size(pyplot, all_data, n_clusters, cluster, probe, profile_dim, size_range, classifier):
    sizes = numpy.zeros(len(DATASET_NAMES))
    corrs = numpy.zeros(len(DATASET_NAMES))
    ps = numpy.zeros(len(DATASET_NAMES))
    for i, dataset in enumerate(DATASET_NAMES):
        data = all_data[(dataset, cluster, n_clusters)][(probe, profile_dim, size_range)]
        curr_sizes = data["size"]
        size = curr_sizes.sum()
        mris = data["mri"][curr_sizes > 0]
        errs = data[classifier][curr_sizes > 0]
        corr, p = pearsonr(mris, errs)
        sizes[i] = size
        corrs[i] = corr
        ps[i] = p
    pyplot.scatter(sizes, corrs, c="k")
    pyplot.axes().set_xscale("log")
    pyplot.savefig("figures/scatter_{}_{}_{}_{:02d}_{:02d}.pdf".format(
        probe, profile_dim, n_clusters, int(100*size_range[0]), int(100*size_range[1])
    ), bbox_inches="tight")
    pyplot.close()


def load_data():
    all_data = {}
    for dataset, cluster_name, n_clusters in product(DATASET_NAMES, CLUSTER_NAMES, CLUSTER_NUMS):
        with open("intermediate/{}.int".format(cluster_file_name(dataset, cluster_name, n_clusters))) as f:
            all_data[(dataset, cluster_name, n_clusters)] = cPickle.load(f)
    return all_data


def synthesis_table(probe, classifier, all_data):
    table_data = prepare_synthesis_table_data(probe, classifier, all_data)
    table_name = "synthesis_{}_{}".format(probe, classifier)
    with open("figures/{}.tex".format(table_name), "w") as f:
        f.writelines((r"\begin{table}\centering", NL, r"\label{tab:%s}" % table_name, NL))
        f.writelines((r"\renewcommand{\arraystretch}{1.1}", NL))
        f.writelines((r"\renewcommand{\tabcolsep}{0pt}", NL))
        f.writelines((r"\begin{tabularx}{0.9\textwidth}{*{3}{>{\small\centering}X}*{5}{W}p{3mm}*{5}{W}}", NL))
        f.writelines((r"\toprule", NL))
        f.writelines((
            r" & & & \multicolumn{5}{c}{%s} & & \multicolumn{5}{c}{%s} \\" % (LEGEND["manual"], LEGEND["kmeans"]), NL,
        ))
        f.writelines((
            r"$\radius_1$ & $\radius_\profiledim$ & $\profiledim$ & ",
            r" & ".join([(r"\multicolumn{1}{c}{$%d$}" % n_c) for n_c in CLUSTER_NUMS]), r" & & ",
            r" & ".join([(r"\multicolumn{1}{c}{$%d$}" % n_c) for n_c in CLUSTER_NUMS]), r" \\",
            NL
        ))
        f.writelines((
            r"\cmidrule(lr){1-3} \cmidrule(lr){4-8} \cmidrule{10-14}"
        ))
        for range_i, table_section in enumerate(table_data):
            f.writelines((
                r"\multirow{5}{*}{%2.0f\%%} & " % (100*SIZE_RANGES[range_i][0]),
                r"\multirow{5}{*}{%2.0f\%%} & " % (100*SIZE_RANGES[range_i][1])
            ))
            for dim_i, dim_data in enumerate(table_section):
                if dim_i > 0:
                    f.write(r" & & ")
                f.write(r"%d & " % PROFILE_DIMS[dim_i])
                for cluster_i, cluster_data in enumerate(dim_data):
                    for n_clusters_i, value in enumerate(cluster_data):
                        value = int(round(value))
                        if value > 66:
                            value = r"\mathbf{%d}" % value
                        f.write("$%s$ " % value)
                        if n_clusters_i < 4 or cluster_i < 1:
                            f.write(r"& ")
                    if cluster_i < 1:
                        f.write(r"& ")
                f.write(r" \\")
                if dim_i < 4:
                    f.write(NL)
            if range_i < 2:
                f.writelines((r"[2mm]", NL))
        f.writelines((r"\bottomrule", NL))
        f.writelines((r"\end{tabularx}", NL))
        caption = r"Percent of datasets on which MRI with {} Probe has correctly estimated {} error rate.".format(
            LEGEND[probe], LEGEND[classifier]
        )
        f.writelines((r"\caption{%s}" % caption, NL))
        f.writelines((r"\end{table}", NL))


def prepare_synthesis_table_data(probe, classifier, all_data):
    n_size_ranges = len(SIZE_RANGES)
    n_profile_dims = len(PROFILE_DIMS)
    n_cluster_names = len(CLUSTER_NAMES)
    n_cluster_nums = len(CLUSTER_NUMS)
    table_data = numpy.zeros((n_size_ranges, n_profile_dims, n_cluster_names, n_cluster_nums))
    for range_i in range(n_size_ranges):
        size_range = SIZE_RANGES[range_i]
        for dim_i in range(n_profile_dims):
            dim = PROFILE_DIMS[dim_i]
            for cluster_i in range(n_cluster_names):
                cluster = CLUSTER_NAMES[cluster_i]
                for n_clusters_i in range(n_cluster_nums):
                    n_clusters = CLUSTER_NUMS[n_clusters_i]
                    for dataset in DATASET_NAMES:
                        mris = all_data[(dataset, cluster, n_clusters)][(probe, dim, size_range)]["mri"]
                        errs = all_data[(dataset, cluster, n_clusters)][(probe, dim, size_range)][classifier]
                        sizes = all_data[(dataset, cluster, n_clusters)][(probe, dim, size_range)]["size"]
                        mris = mris[sizes > 0]
                        errs = errs[sizes > 0]
                        corr, _ = pearsonr(mris, errs)
                        if corr > 0.80:
                            table_data[range_i, dim_i, cluster_i, n_clusters_i] += 1
    table_data = 100 * table_data / len(DATASET_NAMES)
    return table_data


def draw_plot(pyplot, all_data, dataset, cluster, probe, profile_dim, size_range, classifier):
    linestyles = cycle(["-", "--", "-.", ":"])
    markers = cycle(["s", "d", "^", "o"])

    figure_name = "plot_{}_{}_{}_{:02d}_{:02d}_{:02d}_{}".format(
        dataset, cluster, probe, profile_dim,
        int(100*size_range[0]), int(100*size_range[1]),
        classifier
    )

    for n_clusters in CLUSTER_NUMS[1:]:
        data = all_data[(dataset, cluster, n_clusters)][(probe, profile_dim, size_range)]
        mris = data["mri"]
        errs = data[classifier]
        sizes = data["size"]
        mris = mris[sizes > 0]
        errs = errs[sizes > 0]
        pyplot.plot(mris, errs, linestyle=next(linestyles), marker=next(markers),
                    label="{} clusters".format(n_clusters), linewidth=1.5)

    pyplot.legend(loc="upper left")
    pyplot.grid()
    x_range = pyplot.xlim()[1] - pyplot.xlim()[0]
    y_range = pyplot.ylim()[1] - pyplot.ylim()[0]
    pyplot.axes().set_aspect(x_range / y_range)
    pyplot.xticks(numpy.linspace(pyplot.xlim()[0], pyplot.xlim()[1], 6))
    pyplot.yticks(numpy.linspace(pyplot.ylim()[0], pyplot.ylim()[1], 6))

    pyplot.xlabel("MRI ({} Probe)".format(LEGEND[probe]))
    pyplot.ylabel("Error Rate")
    pyplot.title("{} - MRI vs. {} Error Rate".format(
        dataset, LEGEND[classifier]
    ))

    pyplot.savefig("figures/{}.pdf".format(figure_name), bbox_inches="tight")
    pyplot.close()


def configure_matplotlib():
    matplotlib.use('pgf')
    pgf_rc = {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,     # use inline math for ticks
        "pgf.rcfonts": False,    # don't setup fonts from rc parameters
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": [
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{microtype}",
            r"\usepackage{amsfonts}",
            r"\usepackage{amsmath}",
            r"\usepackage{amssymb}",
            r"\usepackage{booktabs}",
            r"\usepackage{fancyhdr}",
            r"\usepackage{graphicx}",
            r"\usepackage{nicefrac}",
            r"\usepackage{xspace}"
        ]
    }
    matplotlib.rcParams.update(pgf_rc)


if __name__ == '__main__':
    main()
