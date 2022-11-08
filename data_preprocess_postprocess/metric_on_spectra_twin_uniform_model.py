import numpy as np
from dataset import dataset_all_2
from model_test_function.model_test_base import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, skew
from scipy.spatial.distance import jensenshannon

"""
This function is used to calculate the performance metrics of models on spectra twin set, 
in order to judge the model's performance on distinguishing spectra twins
The models here are the ones trained on uniform spectra
"""
#%% read data
data_dir='./input/data_nonuniform/file'
label_dir='./input/data_nonuniform/label'
order_dir='./file_reading_order.csv'
spec,temp,mole,_=dataset_all_2(data_dir,label_dir,order_dir)
temp_dif_aug=np.loadtxt('out_save/temp_dif_aug_1e-5_vgg.csv')
#%% filter the data
temp_dif_2=temp_dif_aug[:] # used to analysis slice of data
# temp_dif_sort=temp_dif_2[np.argsort(-temp_dif_2[:,2],)] # check max error
# # temp_dif_sort=temp_dif_2[np.argsort(temp_dif_2[:,5],)] # check min mse
# temp_dif_select=temp_dif_sort[np.where(temp_dif_sort[:,2]>=200)]
temp_dif_select=temp_dif_2[np.where(temp_dif_2[:,5]<=1.0e-5)]
twin_percent=temp_dif_select.shape[0]/temp_dif_2.shape[0]
#### metric calculation
#%% estimation error between estimation and truth
statis_est_spec=all_statistic(temp_dif_select[:,3],temp_dif_select[:,6])
#%%
location_max=np.where(np.abs(temp_dif_select[:,3]-temp_dif_select[:,6])==statis_est_spec[2])
#%% data description of the groundtruth temperature
real_temp=temp_dif_select[:,3]
temp_max=np.max(real_temp)
temp_min=np.min(real_temp)
temp_mean=np.mean(real_temp)
x,bins,patchs=plt.hist(real_temp,bins=10,range=[temp_min,temp_max],weights=np.ones_like(real_temp)/real_temp.shape[0])
plt.show()
#%% between pairs
# # between estimation
statis_est=all_statistic(temp_dif_select[:,6],temp_dif_select[:,7])
# # between pairs
statis_pairs=all_statistic(temp_dif_select[:,3],temp_dif_select[:,4])
#%% relative error
rel_pairs=np.abs((temp_dif_select[:,6]-temp_dif_select[:,7])/(temp_dif_select[:,3]-temp_dif_select[:,4]))
statis_rel_pair=all_statistic_2(rel_pairs)
x, bins, patchs = plt.hist(rel_pairs, bins=10, range=[0, 2.5],
                               weights=np.ones_like(rel_pairs) / rel_pairs.shape[0])
plt.show()
#%% normalized relative err
abs_err=np.abs((temp_dif_select[:,6]-temp_dif_select[:,7]))
normed_abs_pairs=rel_pairs/statis_rel_pair[2]*abs_err/statis_est[4]*abs_err
statis_norm_rel_pair=all_statistic_2(normed_abs_pairs)
# x, bins, patchs = plt.hist(normed_abs_pairs, bins=10, range=[0, 10],
#                                weights=np.ones_like(rel_pairs) / rel_pairs.shape[0])
# plt.show()
skew_norm_pairs=skew(normed_abs_pairs)

#%% the estimatios follow the trend of average temperature change or not
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
#%% test and train index selection
pair_index=temp_dif_select[:,0:2].astype('int')
pair_index_pot=pair_index[7000:]
select_pair=pair_index_pot[np.where(pair_index_pot[:,1]>=7000)]
select_ind_pot=np.reshape(select_pair,[-1])
idex_test=np.unique(select_ind_pot)
# idex_train=np.setdiff1d(temp_dif_select[:,0],idex_test)
train_index_pot=pair_index[:7000,0]
not_good_train=train_index_pot[np.where(pair_index[:7000,1]>=7000)]
idex_train=np.setdiff1d(train_index_pot,not_good_train)