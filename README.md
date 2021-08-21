<h1>Bstrap: A Python Package for confidence intervals and hypothesis testing using bootstrapping.</h1>

You are an **amazing machine learning researcher**.

You invented a new **super cool method**.

You are **not sure that it is significantly better** than your baseline.

You don't have 3000 GPUs to rerun your experiment and check it out.

Then, what you want to do is **bootstrap your results**!

The Bootstrap package allows you to compare two methods and claim that one is better than the other.

## Installation

```bash
pip install bstrap
```
That's all you need, really.

Maybe tough, you can still read the instructions and check out the examples to make sure you get it right...

## Features
Bootstrapping is a simple method to compute statistics over your custom metrics, using only one run of the method for each sample in your evaluation set. It has the advantage of being very versatile, and can be used with any metric really. 

<ul>
  <li>Bootstrapping for computation of confidence interval</li>
  <li>Bootstrapping for hypothesis testing (claim that one method is better than another for a given metric)</li>
  <li>Supports metrics that can be computed sample-wise and metrics that cannot.</li>
</ul>

Keep in mind: non-overlapping confidence intervals means that there is a significant statistical difference. Overlapping confidence intervals does not mean that there is no significant statistical difference. To verify this further, you will need to compute the bootstrap hypothesis testing and check the p-value.

## Instructions

You will need to implement your metric and provide the data sample-wise as a single Pandas dataframe for each method. That's about it.
Your metric is more complex than simply averaging results for each sample? For example, you cannot compute sample-wise, maybe like AUC or mAP? Then just give your predictions and ground truths sample-wise, which also works with Boostrap. 

To use this code, you need to:

<ol>
  <li>Implement your own metric: should take the one pandas dataframe of data as input and return a scalar value.</li>
  <li>Load your data.</li>
  <li>Reformat data to a single pandas dataframe per method with standardized column names, and one sample per row.</li>
  <li>Check that your estimates (confidence interval and p-value) are stable over several runs of the bootstrapping method. If the estimates are not stable, increase nbr_runs</li>
</ol>

Enjoy!

## Usage

You can find example dataframes under src/bstrap/example_dataframes.

#### Example 1: Mean metric 
```python
import pandas as pd
import numpy as np
from bstrap import bootstrap, boostrapping_CI

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
```

#### Example 2: F1 score

```python
import pandas as pd
import sys
from bstrap import bootstrap, boostrapping_CI

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
```

#### Example 3: AUC
```python
import pandas as pd
from sklearn.metrics import auc, roc_curve
from bstrap import bootstrap, boostrapping_CI

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
```

#### Example 4: Multiclass: mean Average Precision (mAP)

```python
import pandas as pd
from sklearn.metrics import roc_curve
from bstrap import bootstrap, boostrapping_CI

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
```

<b>Reference:</b><br/>
Efron, B. and Tibshirani, R.J., 1994. An introduction to the bootstrap. CRC press.
