
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
from dataset import dataset_all, all_statistic

from xgboost import XGBRegressor

from joblib import dump, load

"""
test nonuniform models on ten segment profile data (a more nonuniform profile)
"""
#%% data reading
data_dir='./input/data_10p/file'
label_dir_fake='./input/data_10p/label'
spec,_,__,___=dataset_all(data_dir,label_dir_fake)

label_dir='./input/data_10p/dens_temp_10p.csv'
temp_ave=np.loadtxt(label_dir)

#%% read normalization item
spec_max=np.loadtxt('input/spec_max.csv')
spec_min=np.loadtxt('input/spec_min.csv')
temp_bound=np.loadtxt('input/temp_bound.csv')
#%% data normalization
spec_norm=(spec-spec_min)/(spec_max-spec_min)
#%%
#%% lood gpr
# model=load("model_space/GPR_nonuniform_1e-5.joblib")
#%%
model=XGBRegressor()
model.load_model('model_space/Xgboost_nonuniform_1e-5_new.json')
#%%
temp_pred_norm=model.predict(spec_norm)
temp_pred=temp_pred_norm*(temp_bound[1]-temp_bound[0])+temp_bound[0]
est_statis=all_statistic(temp_ave,temp_pred)
save_dir='output/10p_pred_xgb_1e-5.csv'
np.savetxt(save_dir,temp_pred)

