
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,skew,kurtosis
"""
Calculate all performance metrics for uniform models on uniform data
"""
#%%
pred_dir='predict_test.csv'
test_dir='test_temp.csv'
temp_bound_dir='temp_bound.csv'
#%%
pred_temp_norm=np.loadtxt(pred_dir)
test_temp_norm=np.loadtxt(test_dir)
temp_bound=np.loadtxt(temp_bound_dir)
pred_temp=pred_temp_norm * (temp_bound[1] - temp_bound[0]) + temp_bound[0]
test_temp=test_temp_norm * (temp_bound[1] - temp_bound[0]) + temp_bound[0]
#%% rmse
rmse=np.sqrt(mean_squared_error(pred_temp,test_temp))
abs_err=np.abs(pred_temp-test_temp)
max_err=np.max(abs_err)
min_err=np.min(abs_err)
skew_d=skew(abs_err)
kurt_d=kurtosis(abs_err)
x,bins,patchs=plt.hist(abs_err,bins=10,range=[min_err,max_err],weights=np.ones_like(abs_err)/abs_err.shape[0])
plt.show()
median=np.median(abs_err)
R=pearsonr(pred_temp,test_temp)
perct90=np.percentile(abs_err,75)
#%% original data range
temp_max=np.max(test_temp)
temp_min=np.min(test_temp)
temp_mean=np.mean(test_temp)
x,bins,patchs=plt.hist(test_temp,bins=10,range=[temp_min,temp_max],weights=np.ones_like(test_temp)/test_temp.shape[0])
plt.show()
