from vgg import *
from dataset import *
from sklearn.metrics import mean_squared_error
import torch
import numpy as np
"""
Test network trained on nonuniform profiles on uniform profile data
"""
#%% data reading
data_dir='./input/data_uniform/file'
label_dir='./input/data_uniform/label'
spec,temp_ave,__,___=dataset_all(data_dir,label_dir)


#%% read normalization item
spec_max=np.loadtxt('input/spec_max.csv')
spec_min=np.loadtxt('input/spec_min.csv')
temp_bound=np.loadtxt('input/temp_bound.csv')
#%% data normalization
spec_norm=(spec-spec_min)/(spec_max-spec_min)
temp_norm=(temp_ave-temp_bound[0])/(temp_bound[1]-temp_bound[0])

spec_norm=np.expand_dims(spec_norm,1)
temp_norm=np.expand_dims(temp_norm,1)
test_tcc=torch.from_numpy(spec_norm).float()
#%% load model
model= VGG(make_layers(cfg['B'], batch_norm=False),1)
model.to("cuda")
model_save_dir = './model/vgg_nonuniform_1e-3.pt'
model.load_state_dict(torch.load(model_save_dir))
#%%
test_all=test_tcc[:500].to("cuda")
pred_test_tcc=model(test_all)
pred_test_norm=pred_test_tcc.detach().cpu().numpy().squeeze()
pred_test_norm[np.where(pred_test_norm<0)]=0
pred_test_norm[np.where(pred_test_norm>1)]=1
#%%
pred_test=pred_test_norm*(temp_bound[1]-temp_bound[0])+temp_bound[0]
test_temp_ori=temp_norm*(temp_bound[1]-temp_bound[0])+temp_bound[0]
est_statis=all_statistic(temp_ave[0:500],pred_test)

#%% data check
data_statis=all_statistic_2(temp_ave)
#%%
save_dir="./output/uniform500_pred_vgg_1e-3.csv"
np.savetxt(save_dir,pred_test)