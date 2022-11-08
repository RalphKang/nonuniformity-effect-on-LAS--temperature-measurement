import numpy as np
from dataset import  dataset_all_2, compare_data
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import pearsonr,skew,kurtosis
# from xgboost import XGBRegressor


def single_pair(select_point, temp_dif_select,spec,spec_min,spec_max,temp_bound,model):
    """
    this function is used to select one pair of spectra from time_dif_select matrix, and check
    machine learinng model predicted temp for the pair, and the difference predicted temp with
    original temp

    INPUT:
    select point:the row of time_dif_select;
    temp_dif_select: have explained,the matrix must follow the following format: column 0 is index of main spectrum(spec_location)
    column 1 is index of similar spectrum(location_similar), column 2 is temp difference;
    column 3 is the average temp of main spectrum, column 4 is the average temp of similar spectrum
    spec: list of original spectrum
    spec_min, spec_max: normalization item used when train machine learning model
    temp_bound: temperature normalization bound when train machine learning model
    model: machine learning model used

    OUTPUT:
    a vector which compose of index of main spec, index of similar spec,
    average temp of main spectrum, average temp of simi spectrum,
    predict temp of main spectrum, predicted temp of similar spectrum
    """
    spec_location=int(temp_dif_select[select_point,0]) # first entry is row, second entry is column
    location_similar=int(temp_dif_select[select_point,1])
    temp_truth=temp_dif_select[select_point,3]
    temp_truth_sim=temp_dif_select[select_point,4]

    #%% according to location, find
    spec_select=spec[spec_location]
    spec_select_norm=(spec_select-spec_min)/(spec_max-spec_min)
    spec_select_norm_res=np.reshape(spec_select_norm,[1,-1])

    spec_select_sim=spec[location_similar]
    spec_select_sim_norm=(spec_select_sim-spec_min)/(spec_max-spec_min)
    spec_select_sim_norm_res=np.reshape(spec_select_sim_norm,[1,-1])
    #%%
    temp_pred_norm=model.predict(spec_select_norm_res)
    temp_pred_sim_norm=model.predict(spec_select_sim_norm_res)
    #%%
    temp_pred=temp_pred_norm*(temp_bound[1]-temp_bound[0])+temp_bound[0]
    abs_err=np.abs(temp_pred-temp_truth)

    temp_pred_sim=temp_pred_sim_norm*(temp_bound[1]-temp_bound[0])+temp_bound[0]
    abs_err_sim=np.abs(temp_pred_sim-temp_truth)
    return np.array([spec_location, location_similar,temp_truth,temp_truth_sim,temp_pred[0],temp_pred_sim[0]])


def all_data_comp(temp_dif,spec,spec_min,spec_max,temp_bound,model,save_dir):
    spec_norm = (spec - spec_min) / (spec_max - spec_min)
    temp_pred_norm = model.predict(spec_norm)
    temp_pred = temp_pred_norm * (temp_bound[1] - temp_bound[0]) + temp_bound[0]
    # %%
    temp_main_order = temp_dif[:, 0].astype(np.int)
    temp_pred_main = temp_pred[temp_main_order]

    temp_sim_order = temp_dif[:, 1].astype(np.int)
    temp_pred_sim = temp_pred[temp_sim_order]
    # %%
    temp_dif_temp_pred = np.hstack((temp_dif, np.reshape(temp_pred_main, [-1, 1])))
    temp_dif_aug = np.hstack((temp_dif_temp_pred, np.reshape(temp_pred_sim, [-1, 1])))
    np.savetxt(save_dir,temp_dif_aug)

def abs_max_min(pred_dif_ref, temp_dif_select):
    pred_dif_abs=np.abs(pred_dif_ref)
    max_temp_dif=pred_dif_abs.max() #find the largest predict temperature difference
    min_temp_dif=pred_dif_abs.min()
    max_loc=temp_dif_select[np.where(pred_dif_abs==max_temp_dif)[0]]
    min_loc=temp_dif_select[np.where(pred_dif_abs==min_temp_dif)[0]]
    return max_temp_dif,max_loc,min_temp_dif,min_loc
def all_statistic(data1,data2):
    rmse= np.sqrt(mean_squared_error(data1, data2))
    abs_err = np.abs(data1 - data2)
    max_err = np.max(abs_err)
    min_err = np.min(abs_err)
    median = np.median(abs_err)
    mean=np.mean(abs_err)
    skew_d=skew(abs_err)
    kurt_d=kurtosis(abs_err)
    perct75=np.percentile(abs_err,75)
    R = pearsonr(data1, data2)
    # x, bins, patchs = plt.hist(abs_err, bins=10, range=[min_err, max_err],
    #                            weights=np.ones_like(abs_err) / abs_err.shape[0])
    # plt.show()
    # plt.close()
    return [R[0],rmse,max_err,min_err,median,mean,skew_d,kurt_d,perct75]
def all_statistic_2(abs_err):
    max_err = np.max(abs_err)
    min_err = np.min(abs_err)
    median = np.median(abs_err)
    mean=np.mean(abs_err)
    skew_d = skew(abs_err)
    kurt_d = kurtosis(abs_err)
    perct75 = np.percentile(abs_err, 75)
    # x, bins, patchs = plt.hist(abs_err, bins=10, range=[min_err,max_err],
    #                            weights=np.ones_like(abs_err) / abs_err.shape[0])
    # plt.show()
    # plt.close()
    return [max_err,min_err,median,mean,skew_d,kurt_d,perct75]