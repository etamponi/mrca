import matplotlib
import numpy

__author__ = 'Emanuele Tamponi'


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


from matplotlib import pyplot


inputs = numpy.random.rand(100, 2)

# Random
labels = numpy.random.choice(["o", "d"], size=100)

for x, y in zip(inputs, labels):
    pyplot.scatter([x[0]], [x[1]], c="k", marker=y, alpha=0.5 if y == "d" else 1)

pyplot.xticks([])
pyplot.yticks([])

pyplot.savefig("noisy_neighborhood.pdf", bbox_inches="tight")
pyplot.close()

# Linear boundary

for x in inputs:
    y = "o" if x[0] <= 0.5 else "d"
    pyplot.scatter([x[0]], [x[1]], c="k", marker=y, alpha=0.5 if y == "d" else 1)

pyplot.xticks([])
pyplot.yticks([])

pyplot.savefig("linear_neighborhood.pdf", bbox_inches="tight")
pyplot.close()
