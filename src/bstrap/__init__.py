import pandas as pd
import numpy as np

#RUN WITH PYTHON 3

#Efron, B. and Tibshirani, R.J., 1994. An introduction to the bootstrap. CRC press.
#Bootstrap hypothesis testing

def boostrapping_CI(metric, data ,nbr_runs=1000, verbose=False):

    if verbose:
        print("Computing bootstrap confidence intervals...")

    nbr_scans = len(data.index)
    list_results = []
    # compute metric for each bootstrapped subset
    for r in range(nbr_runs):
        # sample random indexes
        ind = np.random.randint(nbr_scans ,size=nbr_scans)

        # select random subset
        data_bootstrapped = data.iloc[ind]

        # compute metrics
        result = metric(data_bootstrapped)
        list_results.append(result)

    # store variable in dictionary
    metric_stats = dict()
    metric_stats['avg_metric'] = np.average(list_results)
    metric_stats['metric_ci_lb'] = np.percentile(list_results, 5)
    metric_stats['metric_ci_ub'] = np.percentile(list_results, 95)

    if verbose:
        print("Bootstrap confidence intervals computed.")

    return metric_stats


def bootstrap(metric, data_method1, data_method2, nbr_runs=100000, compute_bounds=True, verbose=False):

    # reset index
    data_method1_reindexed = data_method1.reset_index(drop=True)
    data_method2_reindexed = data_method2.reset_index(drop=True)

    # get length of each data
    n = len(data_method1_reindexed.index)
    m = len(data_method2_reindexed.index)
    total = n + m

    # compute the metric for both methods
    result_method1 = metric(data_method1_reindexed)
    result_method2 = metric(data_method2_reindexed)

    # compute statistic t
    t = abs(result_method1 - result_method2)

    # merge data from both methods
    data = pd.concat([data_method1_reindexed, data_method2_reindexed])

    # compute bootstrap statistic
    if verbose:
        print("Computing bootstrap test...")
    nbr_cases_higher = 0
    for r in range(nbr_runs):
        # sample random indexes with replacement
        ind = np.random.randint(total, size=total)

        # select random subset with replacement
        data_bootstrapped = data.iloc[ind]

        # split into two groups
        data_bootstrapped_x = data_bootstrapped[:n]
        data_bootstrapped_y = data_bootstrapped[n:]

        # compute metric for both groups
        result_x = metric(data_bootstrapped_x)
        result_y = metric(data_bootstrapped_y)

        # compute bootstrap statistic
        t_boot = abs(result_x - result_y)

        # compare statistics
        if t_boot > t:
            nbr_cases_higher += 1

    p_value = nbr_cases_higher * 1. / nbr_runs

    if verbose:
        print("Bootstrap test computed.")

    if not compute_bounds:
        return p_value

    else:
        # compute CI and means
        stats_method1 = boostrapping_CI(metric, data_method1, nbr_runs, verbose)
        stats_method2 = boostrapping_CI(metric, data_method2, nbr_runs, verbose)

        return stats_method1, stats_method2, p_value





