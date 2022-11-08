import numpy as np
from dataset import dataset_all_2
from model_test_function.model_test_base import *

"""
this function is used to split spectra twin dataset into training set and test set
"""
#%% read data
spectra_twin_dir='input/temp_dif_data/1e-3/temp_dens_comp.csv'
temp_dif_aug=np.loadtxt(spectra_twin_dir)
#%% filter the data
temp_dif_2=temp_dif_aug[:] # used to analysis slice of data
temp_dif_select=temp_dif_2[np.where(temp_dif_2[:,5]<=1.0e-3)]
#%% test and train index selection
pair_index=temp_dif_select[:,0:2].astype('int')
pair_index_pot=pair_index[7000:]
select_pair=pair_index_pot[np.where(pair_index_pot[:,1]>=7000)]
select_ind_pot=np.reshape(select_pair,[-1])
idex_test=np.unique(select_ind_pot)
train_index_pot=pair_index[:7000,0]
not_good_train=train_index_pot[np.where(pair_index[:7000,1]>=7000)]
idex_train=np.setdiff1d(train_index_pot,not_good_train)
#%% save index
np.savetxt('out_save/spectra_twin_split/nonuniform_train_sample_index_e5_new.csv',idex_train)
np.savetxt('out_save/spectra_twin_split/nonuniform_test_sample_index_e5_new.csv',idex_test)
np.savetxt('out_save/spectra_twin_split/nonuniform_test_pair_e5_new.csv',select_pair)