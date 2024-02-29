"""
Author: Chu-I Yang
Description: Functions for calculating the x- and y-values of uncertainty quantification evaluation plots.

(1) Error-based calibration curve
(2) Confidence-based calibration curve
"""

import scipy
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from sklearn.metrics import mean_squared_error



Array1D = NDArray[np.float64]

def cal_error_based_calibration_metrics(true_arr: Array1D,
                                        pred_arr: Array1D,
                                        unc_arr: Array1D,
                                        n_bins:int = 100) -> Tuple[List, List]:
    """
    The error-based calibration curve examines the consistency between 
    the expected error (measured by mean squared error, MSE) and the predicted uncertainty 
    under the assumption that the estimator is unbiased.

    The error-based calibration curve is a parity plot between the root mean square error (RMSE) and the root mean uncertainty (RMU)

    - Parameters
    true_arr : 1D numpy array of true values in the dataset
    pred_arr : 1D numpy array of predicted values in the dataset
    unc_arr  : 1D numpy array of the predicted uncertainty in the dataset (can be either epistemic, aleatoric or total uncertainty) 
    n_bins   : the number of bins for the dataset (default 100 bins, will calculate 100 points on the plot)
    
    - Returns
    rmu_bins  : x-values of the error-based calibration curve
    rmse_bins : y-values of the error-based calibration curve    
    """
    absolute_error = np.abs(true_arr - pred_arr)

    sorted_matrix = np.vstack((true_arr, pred_arr, unc_arr, absolute_error)).T
    # sort data by uncertainty (predicted variance)
    sorted_matrix = sorted_matrix[np.argsort(sorted_matrix[:, 2].astype(float))]

    rmu_bins = []
    rmse_bins = []
    bin_size = len(true_arr) / n_bins

    for bin_i in range(n_bins):
        start = int(bin_i * bin_size)
        end = int((bin_i+1) * bin_size)
        rmse_bins.append(mean_squared_error(sorted_matrix[start:end, 0], sorted_matrix[start:end, 1], squared=False))
        rmu_bins.append(np.sqrt(np.mean(sorted_matrix[start:end, 2])))

    return rmu_bins, rmse_bins

def cal_confidence_based_calibration_metrics(true_arr: Array1D,
                                             pred_arr: Array1D,
                                             unc_arr: Array1D,
                                             n_bins:int = 10) -> Tuple[List, List]:
    """
    The confidence-based calibration curve examines the fraction of data that actually falls in each confidence level.
    
    - Parameters
    true_arr : 1D numpy array of true values in the dataset
    pred_arr : 1D numpy array of predicted values in the dataset
    unc_arr  : 1D numpy array of the predicted uncertainty in the dataset (can be either epistemic, aleatoric or total uncertainty) 
    n_bins   : the number of bins for the dataset (default 10 bins)
    
    - Returns
    confidence_level : x-values of the error-based calibration curve
    fractions        : y-values of the error-based calibration curve    
    """
    data_size = len(true_arr)
    confidence_level = np.linspace(0, 1, n_bins, endpoint=False)
    
    # the fraction of data that true value falls in the confidence interval
    fractions = []
    for conf in confidence_level:
        count = 0
        for mean, var, true in zip(pred_arr, unc_arr, true_arr):
            lower_bound, upper_bound = scipy.stats.norm.interval(conf, loc=mean, scale=var**0.5)
            if lower_bound < true < upper_bound:
                count += 1
        fractions.append(count/data_size)

    return confidence_level, fractions



    
    
    
