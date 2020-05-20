import os
from scipy.stats import ttest_ind, wilcoxon
import pandas as pd
from ipdb import set_trace as bp
import numpy as np

#RUN WITH PYTHON 3

#Efron, B. and Tibshirani, R.J., 1994. An introduction to the bootstrap. CRC press.
#Bootstrap hypothesis testing

def remove_stats_df(df):

    df = df.drop(index='average',errors='ignore')
    df = df.drop(index='std',errors='ignore')
    df = df.drop(index='95% confidence interval',errors='ignore')
    df = df.drop(index='upper-bound',errors='ignore')
    df = df.drop(index='lower-bound',errors='ignore')

    df = df.transpose()
    df = df.drop(index='average',errors='ignore')
    df = df.drop(index='std',errors='ignore')
    
    df = df.transpose()

    return df

def compute_fauc(meanFpM,meanSensM,maxFP):
    #meanFpM: averaged list of FP per image over all images
    #meanSensM: averaged list of TPR per image over all images
    
    # reverse order for trapz function
    ordFp = np.array(meanFpM[::-1])
    ordSens = np.array(meanSensM[::-1])
    # select all points with less than n FPs
    indsSub = ordFp<maxFP #indices of selected points (boolean array)
    maxSens = np.interp(maxFP,ordFp,ordSens)
    subFp = ordFp[indsSub] #subset FP values selected points
    subFp = np.append(subFp,maxFP) #append maxFP value (last point)
    subSens = ordSens[indsSub] #subset sensitivity rate values selected points
    subSens = np.append(subSens,maxSens) #append interpolated value (last point)
    #sensitivity goes to 1.0, average FP goes to interpolated point
    #calculate partial auc and divide by full surface
    #trapz function calculates auc under curve, Warning: array must be in ascending order! Descending order goes wrong
    fauc = np.trapz(subSens,x=subFp)/np.amax(subFp)  # = auc divided by full surface which is max FP * sensitivity(which is 1.0) = max FP
    
    # Sanity check, highest FP in subset should be interpolated point
    assert np.amax(subFp) == maxFP, "Error, something went wrong with appending!"
   
    
    return fauc

def boostrapping_CI_fauc(FP_matrix,sensitivity_matrix,maxFP):
    #Confidence Interval Estimation of an ROC Curve: An Application of Generalized Half Normal and Weibull Distributions

    nbr_runs = 1000
    nbr_scans = len(FP_matrix.index)
    
    avg_fauc = 0
    var_fauc = 0
    list_fauc = []
    #compute mean
    for r in range(nbr_runs):
        #sample random indexes
        ind = np.random.randint(nbr_scans,size=nbr_scans)
        
        #select random subset
        FP_matrix_bootstrapped = FP_matrix.iloc[ind]
        sensitivity_matrix_bootstrapped = sensitivity_matrix.iloc[ind]

        #compute the means
        meanFpM = FP_matrix_bootstrapped.mean(axis=0)
        meanSensM = sensitivity_matrix_bootstrapped.mean(axis=0)
        
        #compute fauc
        fauc = compute_fauc(meanFpM,meanSensM,maxFP)
        avg_fauc += fauc
        list_fauc.append(fauc)
        
        
    avg_fauc/=nbr_runs
        
    #compute variance
    for r in range(nbr_runs):
        var_fauc += pow(list_fauc[r] - avg_fauc,2)
    var_fauc/=(nbr_runs-1)
    std_fauc=np.sqrt(var_fauc)
    
    #compute CI
    CI = 1.96*std_fauc/np.sqrt(nbr_scans-2)
    
    #store variable in dictionary
    fauc_stats = {}
    fauc_stats['fauc'] = avg_fauc
#    fauc_stats['avg_fauc+CI'] = avg_fauc+CI
#    fauc_stats['avg_fauc-CI'] = avg_fauc-CI
    fauc_stats['fauc_ci_lb'] = np.percentile(list_fauc,5)
    fauc_stats['fauc_ci_ub'] = np.percentile(list_fauc,95)
#    fauc_stats['CI'] = CI
#    fauc_stats['std_fauc'] = std_fauc
#    fauc_stats['avg_fauc+std_fauc'] = avg_fauc+std_fauc
#    fauc_stats['avg_fauc-std_fauc'] = avg_fauc-std_fauc
        
    return fauc_stats, list_fauc

def boostrapping_hypothesisTesting_fauc(maxFP,FP_matrix_1,sensitivity_matrix_1,FP_matrix_2,sensitivity_matrix_2):
    #Confidence Interval Estimation of an ROC Curve: An Application of Generalized Half Normal and Weibull Distributions

    nbr_runs = 100000
    n = len(FP_matrix_1.index)
    m = len(FP_matrix_2.index)
    total = n+m

    #compute the means    
    #compute mean method 1
    meanFpM = FP_matrix_1.mean(axis=0)
    meanSensM = sensitivity_matrix_1.mean(axis=0)       
    #compute fauc
    mean1 = compute_fauc(meanFpM,meanSensM,maxFP)
    #compute mean method 2
    meanFpM = FP_matrix_2.mean(axis=0)
    meanSensM = sensitivity_matrix_2.mean(axis=0)       
    #compute fauc
    mean2 = compute_fauc(meanFpM,meanSensM,maxFP)
    
    #compute statistic t
    t = abs(mean1 - mean2)
    
    #merge
    FP_matrix = pd.concat([FP_matrix_1,FP_matrix_2])
    sensitivity_matrix = pd.concat([sensitivity_matrix_1,sensitivity_matrix_2])
    
    #compute bootstrap statistic
    nbr_cases_higher = 0
    for r in range(nbr_runs):
        #sample random indexes
        ind = np.random.randint(total,size=total)
        
        #select random subset
        FP_matrix_bootstrapped = FP_matrix.iloc[ind]
        sensitivity_matrix_bootstrapped = sensitivity_matrix.iloc[ind]
        
        #select group to compute means
        FP_matrix_bootstrapped_x = FP_matrix_bootstrapped[:n]
        FP_matrix_bootstrapped_y = FP_matrix_bootstrapped[n:]
        sensitivity_matrix_bootstrapped_x = sensitivity_matrix_bootstrapped[:n]
        sensitivity_matrix_bootstrapped_y = sensitivity_matrix_bootstrapped[n:]

        #compute the means
        meanFpM_x = FP_matrix_bootstrapped_x.mean(axis=0)
        meanFpM_y = FP_matrix_bootstrapped_y.mean(axis=0)
        meanSensM_x = sensitivity_matrix_bootstrapped_x.mean(axis=0)
        meanSensM_y = sensitivity_matrix_bootstrapped_y.mean(axis=0)
        
        #compute fauc
        fauc_x = compute_fauc(meanFpM_x,meanSensM_x,maxFP)
        fauc_y = compute_fauc(meanFpM_y,meanSensM_y,maxFP)
        
        #compute bootstrap statistic
        t_boot = abs(fauc_x - fauc_y)
        
        #compare statistics
        if t_boot > t:
            nbr_cases_higher += 1
    
    
    pvalue = nbr_cases_higher*1./nbr_runs
    print(nbr_cases_higher)
    print(pvalue)
    
    return pvalue



if __name__ == '__main__':

    regions = ['0','1','2','3','4','5','6','7','8','9']
    methods = ['simple_gpunet','gpunet','gated','gradCam','backprop','guidedBackprop'] #['simple_gpunet','simple_gpunet_noskip','simple_gpunet_maxpool'] ['simple_gpunet','gpunet','gated','gradCam','backprop','guidedBackprop']
    metrics = ['fauc','FPavg','FNavg','Sensitivity']  #FPavg FNavg Sensitivity fauc
    objective = 'classification' #regression classification arch
    maxFP = 5
    
    folder_results = '7023_mnist_classification_attention_maps_all_stats' #'7026_mnist_regression_attention_maps_gpunet_noskip_maxpool_stats' '7023_mnist_classification_attention_maps_all_stats' '7022_mnist_regression_attention_maps_all_stats'
    
    #result path
    result_path = '/mnt/NASfdubost/homes/fdubost/CloudStation/CloudStation/MNIST_Regression/results'
    
    
    for metric in metrics:
        if metric == 'fauc':
            result_file = 'data_forTest_fauc.csv'
        else:
            result_file = 'data_forTest.csv'
            
        #fill table with all results
        results_test = pd.DataFrame()
        results_test['methods']=methods
        for region in regions:   
            #compute one test per region
            data_forTest = pd.DataFrame()
            
            #fetch result for a single experiment   
            for method in methods:                   
                results = pd.read_csv(os.path.join(result_path,folder_results,region,method,result_file))
                data_forTest[method] = results[metric]
            data_forTest = data_forTest.dropna()    
        
            #find best method (with highest avg)
            best_method = ''
            if metric in ['FPavg','FNavg']:
                best_avg = 10000.
            else:
                best_avg = 0.
            for method in methods:
                method_avg = data_forTest[method].mean()
                if method_avg < best_avg and metric in ['FPavg','FNavg']:
                    best_avg = method_avg
                    best_method = method
                elif method_avg > best_avg and metric in ['Sensitivity','fauc']:
                    best_avg = method_avg
                    best_method = method
            
            #compute wilcoxon test
            results_current_region = {}
            for method in methods:
                if method != best_method:
                    if metric == 'fauc':
                        #load data to compute fauc
                        FP_matrix_best = remove_stats_df(pd.read_csv(os.path.join(result_path,folder_results,region,best_method, 'FP_matrix.csv'),index_col=0))
                        sensitivity_matrix_best = remove_stats_df(pd.read_csv(os.path.join(result_path,folder_results,region,best_method,'sensitivity_matrix.csv'),index_col=0))
                        FP_matrix_method = remove_stats_df(pd.read_csv(os.path.join(result_path,folder_results,region,method, 'FP_matrix.csv'),index_col=0))
                        sensitivity_matrix_method = remove_stats_df(pd.read_csv(os.path.join(result_path,folder_results,region,method,'sensitivity_matrix.csv'),index_col=0))
                        #statistical test
                        pvalue = boostrapping_hypothesisTesting_fauc(maxFP,FP_matrix_best,sensitivity_matrix_best,FP_matrix_method,sensitivity_matrix_method)                    
                    else:
                        pvalue = wilcoxon(data_forTest[best_method],data_forTest[method])[1]
                    results_current_region[method] = '%s' % float('%.2g' % pvalue)
                else:
                    results_current_region[method] = 'best'
            
            #add results for current region to dataframe of results    
            results_test[region]=results_test['methods'].map(results_current_region)
         
        #save results       
        results_test.to_csv(metric+'_results_test_'+objective+'.csv',index=False)






