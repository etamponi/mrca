1)  prepare_predictions and prepare_profiles (they can be run in parallel)
2)  prepare_clusters: 10 clustering files per dataset, each containing the results for 250 profile configurations.

        <dataset>_<cluster_algo>_cluster_<n_clusters>.int

3)  collect_results will collect the results computed in the previous steps in large files.
4)  prepare_figures will prepare all the figures and tables
