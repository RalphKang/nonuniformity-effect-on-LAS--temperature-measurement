### this module is used to contain all the basic function needed to calculate spectra ###
import numpy as np
from math import exp

def light_path_gaussian(temp_low, temp_high, mole_low, mole_high, total_length, node_mag):
    """
     this function is used to generate profiles along the lightpath
    # the needed parameter is temp_low, temp_high, mole_low, mole_high, total_length, node_mag(node distribution)
    # the output is temperature distribution, mole fraction distribution, and column length
    """
    scale_mag = 10  # the sum of length will be divided into more specific grid to get the smooth and exact temp and concentration data

    base = 0  # base is used to count how many columns we have
    node_end = np.zeros_like(node_mag)  # preset for checking the end index of every node.
    for i in range(node_mag.size):  ## used to calculate the end index of every node
        base += node_mag[i]
        node_end[i] = base * scale_mag

    total_node = np.sum(node_mag)
    grid_index = np.arange(0, total_node * scale_mag, 1)  # get a list of grids
    ## set the gaussian profile parameters, the range of
    center = total_node * 5  # Because the scale mag is 10, thus,  use 5 we can get the middle point point of grid
    wing_min = total_node * 2  # the range of wing width to adjust the distribution of gaussian
    wing_max = total_node * 4
    # %%
    wing = wing_min + np.random.rand(1) * (wing_max - wing_min)  # generate random wing parameter
    beta = [exp(-(i - center) ** 2 / wing ** 2) for i in
            grid_index]  # the strength of every index of grid in gaussian distribution

    beta_sec_1 = np.average(beta[:node_end[1]])  # find the strength of middle grid of column 1
    beta_sec_list = [np.average(beta[node_end[i]: node_end[i + 1]]) for i in
                     range(node_mag.size - 1)]  ## find the strength of remained middle grids of columns
    beta_sec_2 = np.array(beta_sec_list)
    beta_sec = np.hstack([beta_sec_1, beta_sec_2])  # combine both, thus get all the middle grids

    temp_dis = beta_sec * (temp_high - temp_low) + temp_low

    # %% calculate the mole concentration and temperature distribution of middle grid.
    wing_mol = wing_min + np.random.rand(1) * (wing_max - wing_min)  # generate random wing parameter
    beta_mol = [exp(-(i - center) ** 2 / wing_mol ** 2) for i in
                grid_index]  # the strength of every index of grid in gaussian distribution

    beta_sec_1_mol = np.average(beta_mol[:node_end[1]])  # find the strength of middle grid of column 1
    beta_sec_list_mol = [np.average(beta_mol[node_end[i]: node_end[i + 1]]) for i in
                         range(node_mag.size - 1)]  ## find the strength of remained middle grids of columns
    beta_sec_2_mol = np.array(beta_sec_list_mol)
    beta_sec_mol = np.hstack([beta_sec_1_mol, beta_sec_2_mol])  # combine both, thus get all the middle grids

    mole_dis = beta_sec_mol * (mole_high - mole_low) + mole_low
    # %% visulization part, not necessary for using, cover it when not needed
    # plt.scatter(node_end, mole_dis)
    # plt.show()
    # plt.scatter(node_end, temp_dis)
    # plt.show()
    # %% the length of each column

    column_length = total_length / total_node * node_mag
    return temp_dis, mole_dis, column_length


def light_path_random(temp_low, temp_high, mole_low, mole_high, total_length, node_mag):
    """
     this function is used to generate profiles along the lightpath follow random distribution
    # the needed parameter is temp_low, temp_high, mole_low, mole_high, total_length, node_mag(node distribution)
    # the output is temperature distribution, mole fraction distribution, and column length
    """
    scale_mag = 10  # the sum of length will be divided into more specific grid to get the smooth and exact temp and concentration data

    base = 0  # base is used to count how many columns we have
    node_end = np.zeros_like(node_mag)  # preset for checking the end index of every node.
    temp_dis = np.zeros_like(node_mag, dtype=np.float32)
    mole_dis = np.zeros_like(node_mag,dtype=np.float32)
    for i in range(node_mag.size):  ## used to calculate the end index of every node
        base += node_mag[i]
        node_end[i] = base * scale_mag
        temp_dis[i] =np.random.rand(1)*(temp_high - temp_low) + temp_low
        mole_dis[i] =np.random.rand(1)*(mole_high - mole_low) + mole_low

    total_node = np.sum(node_mag)
    column_length = total_length / total_node * node_mag
    return temp_dis, mole_dis, column_length
