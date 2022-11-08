
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from dataset import dataset_all, all_statistic
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from joblib import dump, load
"""
test nonuniform models on uniform data
"""

#%% data reading
data_dir='./input/data_uniform/file'
label_dir_fake='./input/data_uniform/label'
spec,temp_ave,__,___=dataset_all(data_dir,label_dir_fake)

# label_dir='./input/data_10p/dens_temp_10p.csv'
# temp_ave=np.loadtxt(label_dir)

#%% read normalization item
spec_max=np.loadtxt('input/spec_max.csv')
spec_min=np.loadtxt('input/spec_min.csv')
temp_bound=np.loadtxt('input/temp_bound.csv')
#%% data normalization
spec_norm=(spec-spec_min)/(spec_max-spec_min)
# %% lood gpr
model=load("model_space/GPR_nonuniform_1e-5.joblib")
#%%
# model=XGBRegressor()
# model.load_model('model_space/Xgboost_nonuniform_1e-5_new.json')
#%%
temp_pred_norm=model.predict(spec_norm)
#constrain prediction range
temp_pred_norm[np.where(temp_pred_norm<0)]=0
temp_pred_norm[np.where(temp_pred_norm>1)]=1
#calculate predicted temperature
temp_pred=temp_pred_norm*(temp_bound[1]-temp_bound[0])+temp_bound[0]
temp_diff=np.abs(temp_ave-temp_pred)
max_dif=np.max(temp_diff)
coord=np.where(temp_diff==max_dif)
pred=temp_pred[coord]
real=temp_ave[coord]
#%%
est_statis=all_statistic(temp_ave,temp_pred)
#%%
save_dir='output/uniform_pred_GPR_1e-5.csv'
np.savetxt(save_dir,temp_pred)

