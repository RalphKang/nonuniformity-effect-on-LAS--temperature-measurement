import numpy as np
from dataset import  dataset_all_2, compare_data
from xgboost import XGBRegressor
from model_test_function.model_test_base import *
from sklearn.gaussian_process import GaussianProcessRegressor
from joblib import load

# def main():
#%% dataset reading
data_dir='./input/data_nonuniform/file'
label_dir='./input/data_nonuniform/label'
order_dir='./file_reading_order.csv'
spec,temp,mole,_=dataset_all_2(data_dir,label_dir, order_dir)
temp_dif=np.loadtxt('input/temp_dif_data/1e-5/temp_dens_comp.csv')
#%% norm
spec_min=np.loadtxt('input/norm_case/spec_min.csv')
spec_max=np.loadtxt('input/norm_case/spec_max.csv')
temp_bound=np.loadtxt('input/norm_case/temp_bound.csv')
#%% data check, random pick some data to check whether temperature can match or not
column=5
if temp_dif[column,1]==np.where(temp_dif[:,3]==temp_dif[column,4])[0]:
    print("true",temp_dif[column,4])
else:
    print('cannot match')
#%%
### model for XGBooster
# model=XGBRegressor()
# model.load_model("input/model_space/Xgboost.json")
### model for GPR
model=load("model/GPR_constant_uniform.joblib")
#%% filter, rank the data according to some priorities
temp_dif_2=temp_dif[:] # used to analysis slice of data
# temp_dif_sort=temp_dif_2[np.argsort(-temp_dif_2[:,2],)] # check max error
# temp_dif_sort=temp_dif_2[np.argsort(temp_dif_2[:,5],)] # check min mse
# temp_dif_select=temp_dif_sort[np.where(temp_dif_sort[:,2]>=200)]
#%% select pairs in data, find their location and their temp data
save_dir='out_save/temp_dif_aug_1e-5_GPR.csv'  # this dir is used to save augmented temp_dif_data

all_data_comp(temp_dif,spec,spec_min,spec_max,temp_bound,model,save_dir)

