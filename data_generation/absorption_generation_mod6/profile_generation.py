import numpy as np
import matplotlib.pyplot as plt
from basement import light_path_random
from hapi_comment import *
import os


# %%
def main():
    node_mag = np.array([1,1,1,1,1,1,1,1,1,1])  # ten segments
    # node_mag=np.array([1]) # uniform profile
    # node_mag=np.array([10, 14, 20, 28, 40])  #five segments it is seperated with the increasement of root of sqr 2
    temp_low = 600
    temp_high = 2000
    total_length = 10.  # cm
    mole_low = 0.05  # mole concentration boundaries, generated from CEA data when temp is 600-2000
    mole_high = 0.07 ## the range set as 0.05 to 0.07 because node ratio is 1.4, to avoid second illpose, maximum mole ratio is 1.4 too.

    # %% spectra setting
    db_begin("HITEMP_500_5000")
    select("co2", DestinationTableName="CO2")

    nu_max = 2395  # (cm-1)
    nu_min = 2375
    stepsize = 0.1
    data_start = 0
    data_amount =500

    file_dir = 'data_save/file'
    img_dir = 'data_save/img'
    label_dir = 'data_save/label'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    for i in range(data_start, data_start + data_amount):
        # %% generate profiles
        temp_dist, mole_dist, column_dist = light_path_random(temp_low, temp_high, mole_low,
                                                       mole_high, total_length, node_mag)
        trans = np.zeros((temp_dist.size, int((nu_max - nu_min) / stepsize + 1)))

        for idx, temp in enumerate(temp_dist):
            wave_number, coef_co2 = absorptionCoefficient_Voigt(SourceTables="CO2", WavenumberRange=(nu_min, nu_max),
                                                                HITRAN_units=False, Environment={'T': temp, 'p': 1},
                                                                mole_fraction=mole_dist[idx], OmegaStep=stepsize)
            _, trans[idx] = transmittanceSpectrum(wave_number, coef_co2,
                                                  Environment={"l": column_dist[idx]})
            # prod_trans*=trans

        final_trans = np.prod(trans, 0)
        final_absorp=1-final_trans

        label_dist = np.vstack((temp_dist, mole_dist))

        file_name = file_dir + '/rand10_' + str(i) + '.csv'
        # np.savetxt(file_name, final_trans)
        np.savetxt(file_name, final_absorp)

        label_name = label_dir + '/rand10_' + str(i) + '.csv'
        np.savetxt(label_name, label_dist)

        # img_name = img_dir + '/' + str(i) + '.png'
        # plt.plot(wave_number, final_absorp)
        # plt.savefig(img_name)
        # plt.close()

        print("finished spectrum generation of {}".format(i))


if __name__ == '__main__':
    main()
