import pandas as pd
import numpy as np

#RUN WITH PYTHON 3

#Efron, B. and Tibshirani, R.J., 1994. An introduction to the bootstrap. CRC press.
#Bootstrap hypothesis testing


def custom_metric(data_methodx):
    metric = 1 #CHANGE
    return metric


def boostrapping_CI(data,nbr_runs=1000):
    #Confidence Interval Estimation of an ROC Curve: An Application of Generalized Half Normal and Weibull Distributions
    
    nbr_scans = len(data.index)
    
    list_metric = []
    #compute mean
    for r in range(nbr_runs):
        #sample random indexes
        ind = np.random.randint(nbr_scans,size=nbr_scans)
        
        #select random subset
        data_bootstrapped = data.iloc[ind]
        
        #compute metrics
        metric = custom_metric(data_bootstrapped)
        list_metric.append(metric)
        
    #store variable in dictionary
    metric_stats = {}
    metric_stats['avg_metric'] = np.average(list_metric) 
    metric_stats['metric_ci_lb'] = np.percentile(list_metric,5)
    metric_stats['metric_ci_ub'] = np.percentile(list_metric,95)

        
    return metric_stats


def boostrapping_hypothesisTesting(data_method1,data_method2,nbr_runs=100000):
    
    n = len(data_method1.index)
    m = len(data_method2.index)
    total = n+m

    #compute the metric for both method    
    metric_method1 = custom_metric(data_method1)
    metric_method2 = custom_metric(data_method2)
    
    #compute statistic t
    t = abs(metric_method1 - metric_method2)
    
    #merge data from both methods
    data = pd.concat([data_method1,data_method2])
    
    #compute bootstrap statistic
    nbr_cases_higher = 0
    for r in range(nbr_runs):
        #sample random indexes with replacement
        ind = np.random.randint(total,size=total)
        
        #select random subset with replacement
        data_bootstrapped = data.iloc[ind]
        
        #split into two groups
        data_bootstrapped_x = data_bootstrapped[:n]
        data_bootstrapped_y = data_bootstrapped[n:]

        #compute metric for both groups
        metric_x = custom_metric(data_bootstrapped_x)
        metric_y = custom_metric(data_bootstrapped_y)
        
        #compute bootstrap statistic
        t_boot = abs(metric_x - metric_y)
        
        #compare statistics
        if t_boot > t:
            nbr_cases_higher += 1
    
    
    pvalue = nbr_cases_higher*1./nbr_runs
    print(nbr_cases_higher)
    print(pvalue)
    
    return pvalue



if __name__ == '__main__':
    
    #You need to:
    #1.implement your own custom_metric function
    #2.change to code to load your data
    #3.check that your estimates (CI bounds and p-value) are stable over several runs of the bootstrapping method. If it is not, increase nbr_runs. 

    #load data
    data_method1 = pd.read_csv("path_data_method1") #CHANGE
    data_method2 = pd.read_csv("path_data_method2") #CHANGE
    
    #compute CI
    metric_stats_method1 = boostrapping_CI(data_method1)
    metric_stats_method2 = boostrapping_CI(data_method2)
    
    #compare method 1 and 2
    pvalue = boostrapping_hypothesisTesting(data_method1,metric_stats_method2) 
    






