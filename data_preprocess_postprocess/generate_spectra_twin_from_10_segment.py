import numpy as np
from dataset import  dataset_all, compare_data
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

"""
This function is used to generate spectra twin from ten segments profile data, if necessary
"""
#%% dataset reading
data_dir='./input/data_10p/file'
label_dir='./input/data_10p/label'
# order_dir='file_reading_order.csv'
spec,temp,mole, data_list=dataset_all(data_dir,label_dir)
#%%
# spec=spec[:200,:]
# temp=temp[:200,:]
# mole=mole[:200,:]
#%%
node_mag = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
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
np.savetxt("out_save/path_temp_10p.csv",temp_path)
np.savetxt("out_save/dens_temp_10p.csv",temp_dens)
#%%
part_stored=False
temp_path_collect=np.zeros([spec.shape[0],6])
temp_dens_collect=np.zeros([spec.shape[0],6])
start_point=0
if part_stored:
    temp_path_collect=np.loadtxt('temp_path_comp.csv')
    temp_dens_collect = np.loadtxt('temp_dens_comp.csv')
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
    np.savetxt('temp_path_comp.csv',temp_path_collect)
    np.savetxt('temp_dens_comp.csv',temp_dens_collect)
# if __name__ == '__main__':
#     main()
#%%
# spec_location=0
# location_similar=45
# #%%
# plt.plot(range(5),temp[spec_location,:],'r')
# plt.plot(range(5),temp[location_similar,:].flatten(),'b')
# plt.ylabel('temperature (K)')
# plt.xlabel('column')
# plt.xlim([0,4])
# plt.legend(['sample0','sample45'])
# plt.show()
# plt.close()
#
# plt.plot(range(5), mole[spec_location, :], 'r')
# plt.plot(range(5), mole[location_similar, :].flatten(), 'b')
# plt.ylabel('mole fraction of CO2')
# plt.xlabel('column')
# plt.xlim([0,4])
# plt.legend(['sample0','sample45'])
# plt.show()
# plt.close()
# #%%
# wavenumber=np.linspace(2975.,2995.,201)
# plt.plot(wavenumber, spec[spec_location, :], 'r')
# plt.plot(wavenumber, spec[location_similar, :].flatten(), 'b')
# plt.ylabel('Absorption spectrum')
# plt.xlabel('wavenumber (cm-1)')
# plt.xlim([2975,2995])
# plt.ylim([0,1])
# plt.legend(['sample0','sample45'])
# plt.show()
# plt.close()