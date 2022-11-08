import numpy as np
from dataset import  dataset_all_2, compare_data
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# def main():
#%% dataset reading
data_dir='./input/data_nonuniform/file'
label_dir='./input/data_nonuniform/label'
order_dir='file_reading_order.csv'
spec,temp,mole, data_list=dataset_all_2(data_dir,label_dir,order_dir)
#%%
node_mag = np.array([10, 14, 20, 28, 40])
node_partition=node_mag/np.sum(node_mag)
#%%
mole_sum=np.sum(mole,1)
mole_sum_res=mole_sum.reshape(-1,1)
mole_norm=mole/mole_sum_res
#%% average path temperature
temp_path=np.dot(temp,node_partition)
#%% average density path temperature
mole_temp_prod=mole_norm*temp
mole_len_prod=np.dot(mole_norm,node_partition)
temp_dens=np.dot(mole_temp_prod,node_partition)/mole_len_prod
#%%
part_stored=False
temp_path_collect=np.zeros([spec.shape[0],6])
temp_dens_collect=np.zeros([spec.shape[0],6])
start_point=0
if part_stored:
    temp_path_collect=np.loadtxt('temp_dif_data/1e-3/temp_path_comp.csv') # average temperature calculated according to path length
    temp_dens_collect = np.loadtxt('temp_dif_data/1e-3/temp_dens_comp.csv') # average temperature calculated considerring concentration
    start_point=10100
for spec_location in range(start_point,spec.shape[0]):
    temp_path_data, temp_dens_data=compare_data(spec,spec_location,threshold=1.0e-3,temp_path=temp_path,temp_dens=temp_dens)
    # ave_temp1=np.dot(temp[spec_location,:],node_partition)
    # ave_temp2=np.dot(temp[(location_similar),:],node_partition)
    temp_path_collect[spec_location]=temp_path_data
    temp_dens_collect[spec_location] = temp_dens_data
    # temp_dif[spec_location,1]=rmse
    # temp_dif[spec_location,2]=spec_location
    # temp_dif[spec_location,3] = location_similar
    print('current location: ',spec_location)
    np.savetxt('temp_dif_data/1e-3/temp_path_comp.csv',temp_path_collect)  # save spectra twins
    np.savetxt('temp_dif_data/1e-3/temp_dens_comp.csv',temp_dens_collect)  # save spectra twins
