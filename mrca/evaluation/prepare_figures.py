from collections import defaultdict
import matplotlib
import numpy
import sys

from mrca.evaluation.collect_results import prepare_data_per_profile, prepare_data_per_clustering, prepare_raw_data
from mrca.evaluation import *


__author__ = 'Emanuele Tamponi'


NL = "\n"


def main():
    configure_matplotlib()
    from matplotlib import pyplot

    raw_data = prepare_raw_data()
    data_per_profile = prepare_data_per_profile()
    data_per_cluster = prepare_data_per_clustering()

    for classifier in CLASSIFIER_NAMES:
        prepare_performance_table(data_per_profile, classifier)
        prepare_mean_correlation_table(data_per_profile, classifier)

    for probe, clusterer, classifier in product(PROBE_NAMES, CLUSTER_NAMES, CLASSIFIER_NAMES):
        best_profile_conf, count = get_best_profile_confs(data_per_profile, 1, probe, clusterer, classifier)[0]
        prepare_profile_table(best_profile_conf, count, data_per_cluster, clusterer, classifier)
    for clusterer, classifier in product(CLUSTER_NAMES, CLASSIFIER_NAMES):
        prepare_best_result_table(data_per_cluster, clusterer, classifier)
        prepare_best_result_plots(pyplot, raw_data, data_per_cluster, clusterer, classifier)


def prepare_performance_table(data_per_profile, classifier):
    table_name = "table_performance_{}".format(classifier)
    with open("figures/{}.tex".format(table_name), "w") as f:
        f.writelines((r"\begin{table}\centering", NL))
        f.writelines((r"\renewcommand{\arraystretch}{1.2}", NL))
        f.writelines((r"\renewcommand{\tabcolsep}{4pt}", NL))
        f.writelines((r"\small", NL))
        f.writelines((
            r"\begin{tabularx}{0.80\textwidth}{*{2}{>{\raggedleft \arraybackslash}X}p{3mm}*{5}{r}p{3mm}*{5}{r}}", NL
        ))
        f.writelines((r"\toprule", NL))
        f.writelines((
            r" & & & \multicolumn{5}{c}{Imbalance Probe} & & \multicolumn{5}{c}{Linear Boundary} \\", NL,
            r"$\countn_1$ & $\countn_\profiledim$ & & 5 & 10 & 15 & 20 & 25 & & 5 & 10 & 15 & 20 & 25 \\", NL,
            r"\midrule", NL
        ))
        for smallest_size in [0.05, 0.10, 0.15, 0.20, 0.25]:
            for largest_size in [0.40, 0.45, 0.50, 0.55, 0.60]:
                size_range = (smallest_size, largest_size)
                if largest_size == 0.50:
                    f.write(r"{}\%".format(int(100*smallest_size)))
                f.write(" & {}\% & ".format(int(100*largest_size)))
                for probe in PROBE_NAMES:
                    for profile_dim in PROFILE_DIMS:
                        corrs = numpy.asarray(
                            [x[1] for x in data_per_profile[(probe, profile_dim, size_range), classifier]]
                        )
                        positive = (corrs >= 0.80).sum()
                        f.write("& {}".format(positive))
                    if probe == "imb":
                        f.write("& ")
                if largest_size == 0.60 and smallest_size < 0.25:
                    f.writelines((r" \\[3mm]", NL))
                else:
                    f.writelines((r" \\", NL))
        f.writelines((
            r"\bottomrule", NL,
            r"\end{tabularx}", NL
        ))
        caption = (
            r"Number of positive results for each profile configuration. Classifier: {}".format(LEGEND[classifier])
        )
        f.writelines((r"\caption{%s}" % caption, NL))
        f.writelines((r"\label{tab:%s}" % table_name, NL))
        f.writelines((r"\end{table}", NL))


def prepare_mean_correlation_table(data_per_profile, classifier):
    table_name = "table_mean_correlation_{}".format(classifier)
    with open("figures/{}.tex".format(table_name), "w") as f:
        f.writelines((r"\begin{table}\centering", NL))
        f.writelines((r"\renewcommand{\arraystretch}{1.2}", NL))
        f.writelines((r"\renewcommand{\tabcolsep}{4pt}", NL))
        f.writelines((r"\small", NL))
        f.writelines((
            r"\begin{tabularx}{0.80\textwidth}{*{2}{>{\raggedleft \arraybackslash}X}p{3mm}*{5}{r}p{3mm}*{5}{r}}", NL
        ))
        f.writelines((r"\toprule", NL))
        f.writelines((
            r" & & & \multicolumn{5}{c}{Imbalance Probe} & & \multicolumn{5}{c}{Linear Boundary} \\", NL,
            r"$\countn_1$ & $\countn_\profiledim$ & & 5 & 10 & 15 & 20 & 25 & & 5 & 10 & 15 & 20 & 25 \\", NL,
            r"\midrule", NL
        ))
        for smallest_size in [0.05, 0.10, 0.15, 0.20, 0.25]:
            for largest_size in [0.40, 0.45, 0.50, 0.55, 0.60]:
                size_range = (smallest_size, largest_size)
                if largest_size == 0.50:
                    f.write(r"{}\%".format(int(100*smallest_size)))
                f.write(" & {}\% & ".format(int(100*largest_size)))
                for probe in PROBE_NAMES:
                    for profile_dim in PROFILE_DIMS:
                        corrs = numpy.asarray(
                            [x[1] for x in data_per_profile[(probe, profile_dim, size_range), classifier]]
                        )
                        positive = 100 * corrs[corrs >= 0.80].mean()**2
                        f.write("& {:.1f}".format(positive))
                    if probe == "imb":
                        f.write("& ")
                if largest_size == 0.60 and smallest_size < 0.25:
                    f.writelines((r" \\[3mm]", NL))
                else:
                    f.writelines((r" \\", NL))
        f.writelines((
            r"\bottomrule", NL,
            r"\end{tabularx}", NL
        ))
        caption = (
            r"Number of positive results for each profile configuration. Classifier: {}".format(LEGEND[classifier])
        )
        f.writelines((r"\caption{%s}" % caption, NL))
        f.writelines((r"\label{tab:%s}" % table_name, NL))
        f.writelines((r"\end{table}", NL))


def prepare_best_result_plots(pyplot, raw_data, data_per_cluster, clusterer, classifier):
    for dataset, n_clusters in product(DATASET_NAMES, CLUSTER_NUMS):
        clustering_conf = (dataset, clusterer, n_clusters)
        profile_conf, corr = data_per_cluster[clustering_conf, classifier][0]
        data = raw_data[clustering_conf, profile_conf]
        draw_plot(pyplot, data, corr, clustering_conf, profile_conf, classifier)


def draw_plot(pyplot, data, corr, clustering_conf, profile_conf, classifier):
    dataset, clusterer, n_clusters = clustering_conf
    figure_name = "best_plot_{}_{}_{}_{}".format(clusterer, n_clusters, classifier, dataset)
    sizes = data["size"]
    mris = data["mri"][sizes > 0]
    errs = data[classifier][sizes > 0]
    pyplot.figure()
    pyplot.gcf().set_size_inches(2, 2)
    pyplot.plot(mris, errs, "ko-")
    x_min, x_max = pyplot.xlim()
    y_min, y_max = pyplot.ylim()
    pyplot.xticks(numpy.linspace(x_min, x_max, 3))
    pyplot.yticks(numpy.linspace(y_min, y_max, 3))
    pyplot.axes().set_aspect((x_max - x_min) / (y_max - y_min))
    pyplot.grid()
    pyplot.title(r"{:.1f}\%{}".format(100*corr*corr, r" $\bullet$" if corr >= 0.8 else ""))
    pyplot.savefig("figures/{}.pdf".format(figure_name), bbox_inches="tight")
    pyplot.close()


def prepare_best_result_table(data_per_cluster, clusterer, classifier):
    table_name = "table_best_results_{}_{}".format(clusterer, classifier)
    totals = defaultdict(int)
    with open("figures/{}.tex".format(table_name), "w") as f:
        f.writelines((r"\begin{table}\centering", NL))
        f.writelines((r"\renewcommand{\arraystretch}{1.1}", NL))
        f.writelines((r"\renewcommand{\tabcolsep}{3pt}", NL))
        f.writelines((r"\small", NL))
        f.writelines((r"\begin{tabularx}{0.80\textwidth}{Xr@{.}l@{}lr@{.}l@{}lr@{.}l@{}lr@{.}l@{}lr@{.}l@{}l}", NL))
        f.writelines((r"\toprule", NL))
        f.writelines((
            r" & \multicolumn{15}{c}{Number of clusters} \\", NL,
            r"\cmidrule{2-16}", NL,
            r"Dataset & \multicolumn{3}{c}{2} & \multicolumn{3}{c}{3} & \multicolumn{3}{c}{4}",
            r" & \multicolumn{3}{c}{5} & \multicolumn{3}{c}{6} \\", NL,
            r"\midrule", NL,
        ))
        for dataset in DATASET_NAMES:
            f.write("{} ".format(dataset))
            for n_clusters in CLUSTER_NUMS:
                corr = data_per_cluster[(dataset, clusterer, n_clusters), classifier][0][1]
                corr_str = "{:.2f}".format(
                    100*corr**2
                ).replace(".", "&")
                if corr >= 0.80:
                    f.write(r"& {} & $\bullet$ ".format(corr_str))
                    totals[n_clusters] += 1
                else:
                    f.write(r"& {} & ".format(corr_str))
            f.writelines((r" \\", NL))
        f.writelines((
            r"\midrule", NL,
            r"Positive results & ", r" & & ".join(
                ("\multicolumn{2}{r}{%d}" % totals[n]) for n in CLUSTER_NUMS
            ), r" & \\", NL,
            r"\bottomrule", NL,
            r"\end{tabularx}", NL
        ))
        caption = (
            r"Best results for each dataset and number of clusters. Clusterer: {}. Compared classifier: {}.".format(
                LEGEND[clusterer], LEGEND[classifier]
            )
        )
        f.writelines((r"\caption{%s}" % caption, NL))
        f.writelines((r"\label{tab:%s}" % table_name, NL))
        f.writelines((r"\end{table}", NL))


def prepare_profile_table(profile_conf, count, data_per_cluster, clusterer, classifier):
    probe, profile_dim, size_range = profile_conf
    table_name = "table_profile_{}_{:02d}_{:02d}_{:02d}_{}_{}".format(
        probe, profile_dim, int(100*size_range[0]), int(100*size_range[1]), clusterer, classifier
    )
    totals = defaultdict(int)
    with open("figures/{}.tex".format(table_name), "w") as f:
        f.writelines((r"\begin{table}\centering", NL))
        f.writelines((r"\renewcommand{\arraystretch}{1.1}", NL))
        f.writelines((r"\renewcommand{\tabcolsep}{3pt}", NL))
        f.writelines((r"\small", NL))
        f.writelines((r"\begin{tabularx}{0.80\textwidth}{Xr@{.}l@{}lr@{.}l@{}lr@{.}l@{}lr@{.}l@{}lr@{.}l@{}l}", NL))
        f.writelines((r"\toprule", NL))
        f.writelines((
            r" & \multicolumn{15}{c}{Number of clusters} \\", NL,
            r"\cmidrule{2-16}", NL,
            r"Dataset & \multicolumn{3}{c}{2} & \multicolumn{3}{c}{3} & \multicolumn{3}{c}{4}",
            r" & \multicolumn{3}{c}{5} & \multicolumn{3}{c}{6} \\", NL,
            r"\midrule", NL,
        ))
        for dataset in DATASET_NAMES:
            f.write("{} ".format(dataset))
            for n_clusters in CLUSTER_NUMS:
                corr = dict(data_per_cluster[(dataset, clusterer, n_clusters), classifier])[profile_conf]
                corr_str = "{:.2f}".format(
                    100*corr**2
                ).replace(".", "&")
                if corr >= 0.80:
                    f.write(r"& {} & $\bullet$ ".format(corr_str))
                    totals[n_clusters] += 1
                else:
                    f.write(r"& {} & ".format(corr_str))
            f.writelines((r" \\", NL))
        f.writelines((
            r"\midrule", NL,
            r"Positive results & ", r" & & ".join(
                ("\multicolumn{2}{r}{%d}" % totals[n]) for n in CLUSTER_NUMS
            ), r" & \\", NL,
            r"\bottomrule", NL,
            r"\end{tabularx}", NL
        ))
        caption = (
            r"Results for {} Probe, \profiledim = {}, $\countn_1 = {}\%$, $\countn_\profiledim = {}\%$. ".format(
            LEGEND[probe], profile_dim, int(100*size_range[0]), int(100*size_range[1]))
        )
        caption += (
            r"Clusterer: {}. Compared classifier: {}.".format(LEGEND[clusterer], LEGEND[classifier])
        )
        f.writelines((r"\caption{%s}" % caption, NL))
        f.writelines((r"\label{tab:%s}" % table_name, NL))
        f.writelines((r"\end{table}", NL))


def get_best_profile_confs(data_per_profile, n, probe, clusterer, classifier):
    print "Best {} {} profile confs".format(n, probe)
    counts = {}
    for profile_conf in PROFILE_CONFS:
        if profile_conf[0] != probe:
            continue
        corrs = numpy.asarray([x[1] for x in data_per_profile[profile_conf, classifier] if x[0][1] == clusterer])
        counts[profile_conf] = (corrs >= 0.80).sum()
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    for conf, count in sorted_counts[:n]:
        print conf, count
    return sorted_counts[:n]


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
