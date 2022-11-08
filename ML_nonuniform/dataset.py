import numpy as np
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr,skew,kurtosis


def dataset_all(data_dir,label_dir):
    """
    this file is used to read all csv files(data and label) in given folder,
     input: data dir, label dir
     output: dataset of data and label

     written by Kang 20211030"""


    data_list=os.listdir(data_dir)

    # for file in data_list:
    #     file1=pd.read_csv(file,header=None)
    data=[np.loadtxt(os.path.join(data_dir,file)) for file in data_list]
    data_np=np.array(data)

    temp=[np.loadtxt(os.path.join(label_dir,file))[0] for file in data_list]
    mole=[np.loadtxt(os.path.join(label_dir,file))[1] for file in data_list]
    temp_np=np.array(temp)
    mole_np=np.array(mole)
    return data_np, temp_np, mole_np, data_list

def dataset_all_2(data_dir,label_dir,order_dir):
    """
    this file is used to read all csv files(data and label) in given folder,
     input: data dir, label dir
     output: dataset of data and label

     written by Kang 20211030"""

    data_list_pd = pd.read_csv('file_reading_order.csv')
    data_list = data_list_pd['0'].values.tolist()

    # for file in data_list:
    #     file1=pd.read_csv(file,header=None)
    data=[np.loadtxt(os.path.join(data_dir,file)) for file in data_list]
    data_np=np.array(data)

    temp=[np.loadtxt(os.path.join(label_dir,file))[0] for file in data_list]
    mole=[np.loadtxt(os.path.join(label_dir,file))[1] for file in data_list]
    temp_np=np.array(temp)
    mole_np=np.array(mole)
    return data_np, temp_np, mole_np, data_list

def compare_data(spec_s_ratio,spec_location,threshold,temp_path,temp_dens):

    spec_1=spec_s_ratio[spec_location]
    spec_del_1 = np.delete(spec_s_ratio, spec_location, 0)

    temp_path_1=temp_path[spec_location]
    temp_path_del_1=np.delete(temp_path, spec_location)
    temp_path_dif=np.abs(temp_path_del_1-temp_path_1)

    temp_dens_1=temp_dens[spec_location]
    temp_dens_del_1=np.delete(temp_dens, spec_location)
    temp_dens_dif=np.abs(temp_dens_del_1-temp_dens_1)
    #%%
    spec_sub = spec_del_1 - spec_1
    spec_sub_sqr = spec_sub ** 2
    mse_np = np.mean(spec_sub_sqr, 1)
    #%%
    mse_min=mse_np.min()

    if mse_min< threshold:
        # mse_select= mse_np[mse_np<=threshold]
        # mse_select_max=mse_select.max()
        # loca=np.where(mse_np==mse_select_max)

        temp_path_select = temp_path_dif[mse_np <= threshold]
        temp_path_select_max = temp_path_select.max()
        loca_path = np.where(temp_path_dif == temp_path_select_max)

        temp_dens_select = temp_dens_dif[mse_np <= threshold]
        temp_dens_select_max = temp_dens_select.max()
        loca_dens = np.where(temp_dens_dif == temp_dens_select_max)
    else:
        # loca=np.where(mse_np==mse_min)
        #
        # mse_select = mse_np[mse_np == mse_min]
        # mse_select_max = mse_select.max()
        # loca = np.where(mse_np == mse_select_max)

        temp_path_select = temp_path_dif[mse_np == mse_min]
        temp_path_select_max = temp_path_select.max() # max difference
        loca_path = np.where(temp_path_dif == temp_path_select_max)

        temp_dens_select = temp_dens_dif[mse_np== mse_min]
        temp_dens_select_max = temp_dens_select.max()
        loca_dens = np.where(temp_dens_dif == temp_dens_select_max)

    loca_max_path = loca_path[0][0]
    loca_max_dens = loca_dens[0][0]
    mse_path = mse_np[loca_max_path]
    mse_dens=mse_np[loca_max_dens]

    if loca_max_path>=spec_location:
        loca_max_path+=1 # if pick before the location_similar,its influence need to consider

    if loca_max_dens >= spec_location:
        loca_max_dens += 1  # if pick before the location_similar,its influence need to consider
    temp_path_final = temp_path[loca_max_path] # path_temperature selected
    temp_dens_final = temp_dens[loca_max_dens]

    path_data_group=np.array([spec_location,loca_max_path,temp_path_select_max,
                              temp_path[spec_location],temp_path_final,mse_path])
    dens_data_group = np.array([spec_location, loca_max_dens, temp_dens_select_max,
                                temp_dens[spec_location], temp_dens_final, mse_dens])
    # print("finish this part")
    #%%
    return path_data_group,dens_data_group

def all_statistic(data1,data2):
    rmse= np.sqrt(mean_squared_error(data1, data2))
    abs_err = np.abs(data1 - data2)
    max_err = np.max(abs_err)
    min_err = np.min(abs_err)
    median = np.median(abs_err)
    mean=np.mean(abs_err)
    skew_d=skew(abs_err)
    kurt_d=kurtosis(abs_err)
    perct75=np.percentile(abs_err,75)
    R = pearsonr(data1, data2)
    # x, bins, patchs = plt.hist(abs_err, bins=10, range=[min_err, max_err],
    #                            weights=np.ones_like(abs_err) / abs_err.shape[0])
    # plt.show()
    # plt.close()
    return [R[0],rmse,max_err,min_err,median,mean,skew_d,kurt_d,perct75]

def all_statistic_2(abs_err):
    max_err = np.max(abs_err)
    min_err = np.min(abs_err)
    median = np.median(abs_err)
    mean=np.mean(abs_err)
    skew_d = skew(abs_err)
    kurt_d = kurtosis(abs_err)
    perct75 = np.percentile(abs_err, 75)
    # x, bins, patchs = plt.hist(abs_err, bins=10, range=[min_err,max_err],
    #                            weights=np.ones_like(abs_err) / abs_err.shape[0])
    # plt.show()
    # plt.close()
    return [max_err,min_err,median,mean,skew_d,kurt_d,perct75]