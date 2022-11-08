import numpy as np
from dataset import dataset_all_2
from model_test_function.model_test_base import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, skew
from scipy.spatial.distance import jensenshannon

"""
This function is used to calculate the performance metrics of models on spectra twin test set, 
in order to judge the model's performance on distinguishing spectra twins
The models here are the ones trained on the nonuniform spectra twin trainiing set
"""
#%% read data
data_dir='./input/data_nonuniform/file'
label_dir='./input/data_nonuniform/label'
order_dir='./file_reading_order.csv'

spec,temp,mole,_=dataset_all_2(data_dir,label_dir,order_dir)
temp_dif_aug=np.loadtxt('out_save/temp_dif_aug_1e-3_xgb.csv')  # 0,spectrum index,1, twin index,2,temp difference
                                                                # 3,spectrum temp,4, twin temp,5, spectrum similarity
                                                                #6,est_temp for spectrum by uniform xgb 7, est temp for twin by uniform xgb
test_dir='./out_save/spectra_twin_split/nonuniform_test_sample_index_e3_new.csv'  # this one is used to load test samples inside the spectra twin test set
test_sample_ind=np.loadtxt(test_dir).astype('int')
pred_temp_of_test_dir='./input/nonuniform_pred_temp/pred_temp_nonuniform_1e-3_new_VGG.csv' # the prediction of temp of whole spectra twin test set
                                                                                           # corresponds to test_sample_ind
pred_temp=np.loadtxt(pred_temp_of_test_dir)
test_pair_dir='./out_save/spectra_twin_split/nonuniform_test_pair_e3_new.csv'  # the index of spectra and labels inside the test set
test_pair=np.loadtxt(test_pair_dir)
#%%
if pred_temp.shape[0]!=test_sample_ind.shape[0]:
    print("size of predition index are not equal to size of prediction temp")



#%% filter the data
temp_dif_2=temp_dif_aug[:] # used to analysis slice of data
# temp_dif_sort=temp_dif_2[np.argsort(-temp_dif_2[:,2],)] # check max error
# # temp_dif_sort=temp_dif_2[np.argsort(temp_dif_2[:,5],)] # check min mse
# temp_dif_select=temp_dif_sort[np.where(temp_dif_sort[:,2]>=200)]
pair_need=test_pair[:,0].astype('int')
temp_dif_select=temp_dif_2[pair_need]
# twin_percent=temp_dif_select.shape[0]/temp_dif_2.shape[0]
#%%
ind_predtemp_pair=np.hstack((test_sample_ind.reshape([-1,1]),pred_temp.reshape([-1,1])))
#%%
temp_pred=np.array([ind_predtemp_pair[np.where(ind_predtemp_pair[:,0]==temp_dif_select[i,0])[0],1] for i in range(temp_dif_select.shape[0])])
twin_temp_pred=np.array([ind_predtemp_pair[np.where(ind_predtemp_pair[:,0]==temp_dif_select[i,1])[0],1] for i in range(temp_dif_select.shape[0])])

temp_dif_select[:,6]=temp_pred.squeeze()
temp_dif_select[:,7]=twin_temp_pred.squeeze()
#### metric calculation
#%% estimation error between estimation and truth
statis_est_spec=all_statistic(temp_dif_select[:,3],temp_dif_select[:,6])
#%%
pos_max_abs_err=np.where(np.abs(temp_dif_select[:,3]-temp_dif_select[:,6])==statis_est_spec[2])
#%% data description of the groundtruth temperature
real_temp=temp_dif_select[:,3]
data_desc=all_statistic_2(real_temp)
#%% between pairs
# # between estimation
statis_est=all_statistic(temp_dif_select[:,6],temp_dif_select[:,7])
# # between pairs
statis_pairs=all_statistic(temp_dif_select[:,3],temp_dif_select[:,4])
#%% relative error
rel_pairs=np.abs((temp_dif_select[:,6]-temp_dif_select[:,7])/(temp_dif_select[:,3]-temp_dif_select[:,4]))
statis_rel_pair=all_statistic_2(rel_pairs)
# x, bins, patchs = plt.hist(rel_pairs, bins=10, range=[0, 2.5],
#                                weights=np.ones_like(rel_pairs) / rel_pairs.shape[0])
# plt.show()
#%% normalized relative err
abs_err=np.abs((temp_dif_select[:,6]-temp_dif_select[:,7]))
normed_abs_pairs=rel_pairs/statis_rel_pair[2]*abs_err/statis_est[4]*abs_err
statis_norm_rel_pair=all_statistic_2(normed_abs_pairs)
# x, bins, patchs = plt.hist(normed_abs_pairs, bins=10, range=[0, 10],
#                                weights=np.ones_like(rel_pairs) / rel_pairs.shape[0])
# plt.show()
skew_norm_pairs=skew(normed_abs_pairs)

#%% percentage of consistency
rel_no_abs=(temp_dif_select[:,6]-temp_dif_select[:,7])/(temp_dif_select[:,3]-temp_dif_select[:,4])
count_distrend=np.where(rel_no_abs<0.0)[0]
percentage_not_follow_trend=count_distrend.shape[0]/rel_pairs.shape[0]
percentage_follow_trend=1-percentage_not_follow_trend
#%% percentage of closeness
close_metric=np.abs((temp_dif_select[:,6]-temp_dif_select[:,3])/(temp_dif_select[:,6]-temp_dif_select[:,4]))
close_satisfy=close_metric[np.where(close_metric<1.)].shape[0]/rel_pairs.shape[0]
#%% percentage of middleness
middle_metric=(temp_dif_select[:,6]-temp_dif_select[:,3])/(temp_dif_select[:,6]-temp_dif_select[:,4])
middle_satisfy=middle_metric[np.where(middle_metric<0.)].shape[0]/rel_pairs.shape[0]
#%% JS Divergence
abs_err_no_ab=temp_dif_select[:,6]-temp_dif_select[:,7]
abs_err_norm=(abs_err_no_ab-abs_err_no_ab.min())/(abs_err_no_ab.max()-abs_err_no_ab.min())
abs_err_pair=temp_dif_select[:,3]-temp_dif_select[:,4]
abs_err_pair_norm=(abs_err_pair-abs_err_pair.min())/(abs_err_pair.max()-abs_err_pair.min())
dist_abs_err=np.histogram(abs_err_norm,bins=30,range=[0,1],
                          weights=np.ones_like(abs_err_norm)/abs_err_norm.shape[0])
dist_abs_err_pair=np.histogram(abs_err_pair_norm,bins=30,range=[0,1],
                               weights=np.ones_like(abs_err_pair_norm)/abs_err_pair_norm.shape[0])
js_diverg=jensenshannon(dist_abs_err[0],dist_abs_err_pair[0])
