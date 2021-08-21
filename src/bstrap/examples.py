import pandas as pd
import numpy as np
import sys
from sklearn.metrics import auc, average_precision_score, roc_curve
from bstrap import bootstrap, boostrapping_CI
from ipdb import set_trace as bp

#RUN WITH PYTHON 3

#Efron, B. and Tibshirani, R.J., 1994. An introduction to the bootstrap. CRC press.
#Bootstrap hypothesis testing

if __name__ == '__main__':
    # INSTRUCTIONS
    # You need to:
    # 1. Implement your own metric: should take the one pandas dataframe of data as input and return a scalar value.
    # 2. Load your data
    # 3. Reformat data to a single pandas dataframe per method with standardized column names, and one sample per row.
    # 3. Check that your estimates (CI bounds and p-value) are stable over several runs of the bootstrapping method.
    # If the estimates are not stable, increase nbr_runs.

    # EXAMPLE 1: MEAN METRIC ------------------------------------------------------------------------------------------
    # 1. implement metric
    metric = np.mean

    # 2. load data
    df = pd.read_csv("example_dataframes/example_dataframe_mean.csv")

    # 3. reformat data to a single pandas dataframe per method
    data_method1 = df["loss_method_1"]
    data_method2 = df["loss_method_2"]

    # 4. compare method 1 and 2
    stats_method1, stats_method2, p_value = bootstrap(metric, data_method1, data_method2, nbr_runs=1000)
    print(stats_method1)
    print(stats_method2)
    print(p_value)

    # compute CI and mean for each method separately
    stats_method1 = boostrapping_CI(metric, data_method1, nbr_runs=1000)
    stats_method2 = boostrapping_CI(metric, data_method2, nbr_runs=1000)
    print(stats_method1)
    print(stats_method2)

    # EXAMPLE 2: F1 score ---------------------------------------------------------------------------------------------
    # 1. implement metric
    def compute_f1(data):
        val_target = data["gt"].astype('bool')
        val_predict = data["predictions"].astype('bool')
        tp = np.count_nonzero(val_target * val_predict)
        fp = np.count_nonzero(~val_target * val_predict)
        fn = np.count_nonzero(val_target * ~val_predict)
        return tp * 1. / (tp + 0.5 * (fp + fn) + sys.float_info.epsilon)
    metric = compute_f1

    # 2. load data
    df = pd.read_csv("example_dataframes/example_dataframe_f1.csv")

    # 3. reformat data to a single pandas dataframe per method with standardized column names
    data_method1 = df[["gt", "method1"]]
    data_method1 = data_method1.rename(columns={"method1": "predictions"})
    data_method2 = df[["gt", "method2"]]
    data_method2 = data_method2.rename(columns={"method2": "predictions"})

    # 4. compare method 1 and 2 (same code as example 1)
    stats_method1, stats_method2, p_value = bootstrap(metric, data_method1, data_method2, nbr_runs=1000)
    print(stats_method1)
    print(stats_method2)
    print(p_value)

    # compute CI and mean for each method separately (same code as example 1)
    stats_method1 = boostrapping_CI(metric, data_method1, nbr_runs=1000)
    stats_method2 = boostrapping_CI(metric, data_method2, nbr_runs=1000)
    print(stats_method1)
    print(stats_method2)

    # EXAMPLE 3: AUC --------------------------------------------------------------------------------------------------
    # 1. implement metric
    def compute_auc(data):
        fpr, tpr, thresholds = roc_curve(data["gt"], data["predictions"], pos_label=1)
        return auc(fpr, tpr)
    metric = compute_auc

    # 2. load data
    df = pd.read_csv("example_dataframes/example_dataframe_auc.csv")

    # 3. reformat data to a single pandas dataframe per method with standardized column names
    data_method1 = df[["gt", "method1"]]
    data_method1 = data_method1.rename(columns={"method1": "predictions"})
    data_method2 = df[["gt", "method2"]]
    data_method2 = data_method2.rename(columns={"method2": "predictions"})

    # 4. compare method 1 and 2 (same code as example 1)
    stats_method1, stats_method2, p_value = bootstrap(metric, data_method1, data_method2, nbr_runs=1000)
    print(stats_method1)
    print(stats_method2)
    print(p_value)

    # compute CI and mean for each method separately (same code as example 1)
    stats_method1 = boostrapping_CI(metric, data_method1, nbr_runs=1000)
    stats_method2 = boostrapping_CI(metric, data_method2, nbr_runs=1000)
    print(stats_method1)
    print(stats_method2)

    # EXAMPLE 4: MULTICLASS: MEAN AVERAGE PRECISION (mAP) -------------------------------------------------------------
    # 1. implement metric
    def compute_mAP(data):
        gt = data[[column for column in data.columns if 'gt' in column]]
        predictions = data[[column for column in data.columns if 'pred' in column]]
        return average_precision_score(gt, predictions, average='weighted')
    metric = compute_mAP

    # 2. load data
    gt = pd.read_csv("example_dataframes/example_dataframe_mAP_gt.csv")
    predictions_method1 = pd.read_csv("example_dataframes/example_dataframe_mAP_predictions_method1.csv")
    predictions_method2 = pd.read_csv("example_dataframes/example_dataframe_mAP_predictions_method2.csv")

    # 3. reformat data to a single pandas dataframe per method with standardized column names
    gt = gt.rename(columns=dict([(column, 'gt_' + column) for column in gt.columns]))
    predictions_method1 = predictions_method1.rename(
        columns=dict([(column, 'pred_' + column) for column in predictions_method1.columns]))
    predictions_method2 = predictions_method2.rename(
        columns=dict([(column, 'pred_' + column) for column in predictions_method2.columns]))
    data_method1 = pd.concat([gt, predictions_method1], axis=1)
    data_method2 = pd.concat([gt, predictions_method2], axis=1)

    # 4. compare method 1 and 2 (same code as example 1)
    stats_method1, stats_method2, p_value = bootstrap(metric, data_method1, data_method2, nbr_runs=100)
    print(stats_method1)
    print(stats_method2)
    print(p_value)

    # compute CI and mean for each method separately (same code as example 1)
    stats_method1 = boostrapping_CI(metric, data_method1, nbr_runs=100)
    stats_method2 = boostrapping_CI(metric, data_method2, nbr_runs=100)
    print(stats_method1)
    print(stats_method2)
