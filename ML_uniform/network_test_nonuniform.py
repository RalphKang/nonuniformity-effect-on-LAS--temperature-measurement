from network_archive.vgg import *
from dataset import dataset_all, dataset_all_2
from sklearn.metrics import mean_squared_error
import torch
import numpy as np

"""
This code is used to test network models on nonuniform twins, and
 record the predict temperature in temp_dif_aug_1e-x_xxx.csv
"""
#%% data reading
data_dir='./input/data_nonuniform/file'
label_dir='./input/data_nonuniform/label'
order_dir='./file_reading_order.csv'
spec,_,__,___=dataset_all_2(data_dir,label_dir, order_dir)
"""
read spectra twins, the information inside the temp_dens_comp.csv are as follows:
first column: 0,index of current spectra
second column: 1, the counrterpart index to make up spectra twin
third column: 2, the temperature difference
fourth column: 3, the average temperature for current spectra
fifth column: 4, the average temperature for the counterpart spectra
sixth column: 5, the similarity level
"""
temp_dif=np.loadtxt('input/temp_dif_data/1e-3/temp_dens_comp.csv')
#%% data check, random pick some data to check whether temperature can match or not
column=5
if temp_dif[column,1]==np.where(temp_dif[:,3]==temp_dif[column,4])[0]:
    print("true",temp_dif[column,4])
else:
    print('cannot match')

#%% read normalization item
spec_max=np.loadtxt('input/norm_case/spec_max.csv')
spec_min=np.loadtxt('input/norm_case/spec_min.csv')
temp_bound=np.loadtxt('input/norm_case/temp_bound.csv')
#%% data normalization
spec_norm=(spec-spec_min)/(spec_max-spec_min)

spec_norm=np.expand_dims(spec_norm,1)
test_tcc=torch.from_numpy(spec_norm).float()
#%% load model
model= VGG(make_layers(cfg['B'], batch_norm=False),1)
model.to("cuda")
model_save_dir = './model/vgg_B_uniform.pt'
model.load_state_dict(torch.load(model_save_dir))
#%%

pred_test_norm=[]
for test in test_tcc:
    test_res=test.reshape([1,1,-1])
    test_res=test_res.to('cuda')
    pred_test=model(test_res)
    pred_test_norm.append(pred_test.detach().cpu().numpy().squeeze())
#%%
pred_test_norm=np.array(pred_test_norm)
#%%
temp_pred=pred_test_norm*(temp_bound[1]-temp_bound[0])+temp_bound[0]

temp_main_order = temp_dif[:, 0].astype(np.int)
temp_pred_main = temp_pred[temp_main_order]

temp_sim_order = temp_dif[:, 1].astype(np.int)
temp_pred_sim = temp_pred[temp_sim_order]
# %%
temp_dif_temp_pred = np.hstack((temp_dif, np.reshape(temp_pred_main, [-1, 1])))
temp_dif_aug = np.hstack((temp_dif_temp_pred, np.reshape(temp_pred_sim, [-1, 1])))

#%%
save_dir='out_save/temp_dif_aug_1e-3_vgg.csv'
np.savetxt(save_dir, temp_dif_aug)
#%%
# np.savetxt('./input/data_10p/dens_temp_10p_pred.csv',pred_test)