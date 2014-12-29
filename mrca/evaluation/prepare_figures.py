from collections import defaultdict
import matplotlib
import numpy

from mrca.evaluation.collect_results import prepare_data_per_profile, prepare_data_per_clustering
from mrca.evaluation import *


__author__ = 'Emanuele Tamponi'


NL = "\n"


def main():
    # configure_matplotlib()
    from matplotlib import pyplot

    data_per_profile = prepare_data_per_profile()
    data_per_cluster = prepare_data_per_clustering()

    for probe, clusterer, classifier in product(PROBE_NAMES, CLUSTER_NAMES, CLASSIFIER_NAMES):
        best_profile_conf, count = get_best_profile_confs(data_per_profile, 1, probe, clusterer, classifier)[0]
        prepare_profile_table(best_profile_conf, count, data_per_cluster, clusterer, classifier)


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
                )
                if corr >= 0.80:
                    f.write(r"& {} & $\bullet$ ".format(corr_str.replace(".", "&")))
                    totals[n_clusters] += 1
                else:
                    f.write(r"& {} & ".format(corr_str.replace(".", "&")))
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
