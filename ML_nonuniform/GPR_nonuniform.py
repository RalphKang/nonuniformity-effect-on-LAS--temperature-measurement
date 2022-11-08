from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error

from dataset import dataset_all_2
"""
Train GPR model on different similarity level dataset
"""
# def main():
#%% generate dataset
data_dir='./input/data_nonuniform/file'
label_dir='./input/data_nonuniform/label'
order_dir='./file_reading_order.csv'
spec,temp,mole,_=dataset_all_2(data_dir,label_dir,order_dir)

temp_dif_dir='./input/temp_dif_data/1e-3/temp_dens_comp.csv'
temp_dif_select=np.loadtxt(temp_dif_dir)

spec_max=np.loadtxt('input/spec_max.csv')
spec_min=np.loadtxt('input/spec_min.csv')
temp_bound=np.loadtxt('input/temp_bound.csv')

train_index_dir='./input/nonuniform_train_sample_index_e3_new.csv'
test_index_dir='./input/nonuniform_test_sample_index_e3_new.csv'
train_index=np.loadtxt(train_index_dir)
test_index=np.loadtxt(test_index_dir)
# spec=np.log(spec)
#%%
temp_ave=temp_dif_select[:,3]
#%% data_normalization
spec_norm=(spec-spec_min)/(spec_max-spec_min)
temp_norm=(temp_ave-temp_bound[0])/(temp_bound[1]-temp_bound[0])
#%% split dataset
train_ind_int=train_index.astype(int)
test_ind_int=test_index.astype(int)

train_spec=spec_norm[train_ind_int]
test_spec=spec_norm[test_ind_int]

train_temp=temp_norm[train_ind_int]
test_temp=temp_norm[test_ind_int]
#%% define model
# model=SVR(gamma='scale',kernel="poly").fit(train_spec, train_temp)
model=GaussianProcessRegressor().fit(train_spec, train_temp)
#%%
temp_pred_test=model.predict(test_spec)
temp_pred_train=model.predict(train_spec)
#%% judge model
# mse=mean_squared_error(test_temp[:,column],temp_pred)
mse_test=mean_squared_error(test_temp,temp_pred_test)
mse_train=mean_squared_error(train_temp,temp_pred_train)

#%%
temp_max=temp_bound[1]
temp_min=temp_bound[0]
mse_test_ori=mean_squared_error(test_temp*(temp_max-temp_min)+temp_min,temp_pred_test*(temp_max-temp_min)+temp_min)
mse_train_ori=mean_squared_error(train_temp*(temp_max-temp_min)+temp_min,temp_pred_train*(temp_max-temp_min)+temp_min)
#%%
np.savetxt('output/pred_temp_nonuniform_1e-3_new_GPR.csv',temp_pred_test*(temp_max-temp_min)+temp_min)

#%%
dump(model,"model_space/GPR_nonuniform_1e-3.joblib")