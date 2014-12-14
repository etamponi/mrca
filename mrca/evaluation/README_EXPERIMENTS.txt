1)  Run prepare_predictions and prepare_profiles (they can be run in parallel, but set the number of processes
    in the file if you have low RAM)
2)  Run prepare_clusters to have the final results. Analysis for each dataset is in the files:

        <dataset>_<cluster_algo>_<n_clusters>

    and there are 36(datasets) x 30(profile configurations) x 2(cluster algorithms) x 5(numbers of clusters) results
3)  analyze_results will prepare all the figures and tables
