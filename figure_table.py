from mrca.evaluation import DATASET_NAMES

__author__ = 'Emanuele Tamponi'


s = r"""
  \begin{subfigure}{0.24\textwidth}
    \includegraphics[width=\textwidth]
    {Figures/Chapter2/best_plot_manual_3_ab_%s.pdf}
    \caption*{%s, 3 c.}
  \end{subfigure}%%
  \begin{subfigure}{0.24\textwidth}
    \includegraphics[width=\textwidth]
    {Figures/Chapter2/best_plot_manual_4_ab_%s.pdf}
    \caption*{%s, 4 c.}
  \end{subfigure}%%
  \begin{subfigure}{0.24\textwidth}
    \includegraphics[width=\textwidth]
    {Figures/Chapter2/best_plot_manual_5_ab_%s.pdf}
    \caption*{%s, 5 c.}
  \end{subfigure}%%
  \begin{subfigure}{0.24\textwidth}
    \includegraphics[width=\textwidth]
    {Figures/Chapter2/best_plot_manual_6_ab_%s.pdf}
    \caption*{%s, 6 c.}
  \end{subfigure}%%
"""


for dataset in DATASET_NAMES:
    print s % (dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset),
