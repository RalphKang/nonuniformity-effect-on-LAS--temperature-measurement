import numpy as np
from joblib import dump
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from xgboost import XGBRegressor

from dataset import dataset_all, all_statistic
"""
This code is used to train machine learning models for uniform profile spectra measurement;
The uniform spectra and labels are stored in data_save/uniform_spectra
the trained models are saved in model_space
the metrics about machine learning models are saved in output folder, the metrics include R, rmse, max_err, min_err, 
median, mean, skew_d, kurt_d, perct75, please check more details in all_statistics
"""
#%% generate dataset
data_dir='./input/data_uniform/file'
label_dir='./input/data_uniform/label'
spec,temp,mole=dataset_all(data_dir,label_dir)

#%%
data_size=temp.shape
data_length=data_size[0]
train_length=int(data_length*0.9)
train_spec=spec[:train_length]
test_spec=spec[train_length:]

spec_max=train_spec.max(0)
spec_min=train_spec.min(0)
spec_norm_train=(train_spec-spec_min)/(spec_max-spec_min)
spec_norm_test =(test_spec-spec_min)/(spec_max-spec_min)

temp_max=temp.max(0)
temp_min=temp.min(0)
temp_norm=(temp-temp_min)/(temp_max-temp_min)

train_temp=temp_norm[:train_length]
test_temp=temp_norm[train_length:]

#%% define model, if you want to train SVR or Gausssian model, please uncomment them and use the savefile for sklearn
# model=SVR(gamma='scale')
# model.fit(train_spec,train_temp)
# model = SVR(gamma='scale',kernel="precomputed").fit(spec_norm_train, train_temp)  # svr model

kernel=RBF()
model = GaussianProcessRegressor(kernel=kernel).fit(spec_norm_train, train_temp) # gaussian process model

# kernel=RBF()
# model = KernelRidge(kernel="rbf").fit(spec_norm_train, train_temp)  # ridge model
# store model
model_dir='model/GPR_rbf_uniform.joblib'
dump(model,model_dir)
#%% Xgboost model
# model = XGBRegressor(n_estimators=100, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8,num_parallel_tree=100).fit(spec_norm_train, train_temp)
# model_dir='model/xgboost_uniform.json'
# model.save_model(model_dir)
#%% prediction from models
temp_test_pred=model.predict(spec_norm_test)
temp_train_pred=model.predict(spec_norm_train)
#%% data denormalization
temp_train_orig=temp_train_pred*(temp_max-temp_min)+temp_min
temp_train_pred_orig=temp_train_pred*(temp_max-temp_min)+temp_min

temp_test_orig=test_temp*(temp_max-temp_min)+temp_min
temp_test_pred_orig=temp_test_pred*(temp_max-temp_min)+temp_min
#%% calculation metrics
metric_test=all_statistic(temp_test_orig,temp_test_pred_orig)
metric_train=all_statistic(temp_train_orig,temp_train_pred_orig)
metric_train_test=np.vstack((metric_train,metric_test))
#%% record metrics
metric_dir='output/gpr_rbf_metric.csv'
np.savetxt(metric_dir,metric_train_test)
prediction_dir="output/predict_temp_uniform_GPR.csv"
np.savetxt(prediction_dir,temp_test_pred_orig)


