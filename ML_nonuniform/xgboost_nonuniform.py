import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from dataset import dataset_all_2
"""
Train GPR on different similarity dataset
"""
# def main():
#%% generate dataset
data_dir='./input/data_nonuniform/file'
label_dir='./input/data_nonuniform/label'
order_dir='./file_reading_order.csv'
spec,temp,mole,_=dataset_all_2(data_dir,label_dir,order_dir)

temp_dif_dir='./input/temp_dif_data/1e-5/temp_dens_comp.csv'
temp_dif_select=np.loadtxt(temp_dif_dir)

spec_max=np.loadtxt('input/spec_max.csv')
spec_min=np.loadtxt('input/spec_min.csv')
temp_bound=np.loadtxt('input/temp_bound.csv')

train_index_dir='./input/nonuniform_train_sample_index_e5_new.csv'
test_index_dir='./input/nonuniform_test_sample_index_e5_new.csv'
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
#%%
train_spec=spec[train_ind_int]
test_spec=spec[test_ind_int]

train_temp=temp_norm[train_ind_int]
test_temp=temp_norm[test_ind_int]
model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8).fit(train_spec, train_temp)
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
np.savetxt('output/pred_temp_nonuniform_1e-5_new_xgb.csv',temp_pred_test*(temp_max-temp_min)+temp_min)
#%%
model.save_model('model_space/Xgboost_nonuniform_1e-5_new.json')
np.savetxt('spec_min.csv',spec_min)
np.savetxt('spec_max.csv',spec_max)
np.savetxt('temp_bound.csv',np.array([temp_min, temp_max]))