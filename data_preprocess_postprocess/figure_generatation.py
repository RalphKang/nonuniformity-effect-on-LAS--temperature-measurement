import numpy as np
from dataset import dataset_all_2
from model_test_function.model_test_base import *
import matplotlib.pyplot as plt
#%%
data_dir='./input/data_nonuniform/file'
label_dir='./input/data_nonuniform/label'
order_dir='./file_reading_order.csv'
spec,temp,mole,_=dataset_all_2(data_dir,label_dir,order_dir)
nonuniform_perf_dir='input/nonuniform_pred_spectra_twin/temp_dif_aug_1e-4_vgg.csv'
temp_dif_aug=np.loadtxt(nonuniform_perf_dir)

test_dir='./out_save/spectra_twin_split/nonuniform_test_sample_index_e4_new.csv'
test_sample_ind=np.loadtxt(test_dir).astype('int')
pred_temp_of_test_dir='./input/nonuniform_pred_temp/pred_temp_nonuniform_1e-4_new_VGG.csv'
pred_temp=np.loadtxt(pred_temp_of_test_dir)
test_pair_dir='./out_save/spectra_twin_split/nonuniform_test_pair_e4_new.csv'
test_pair=np.loadtxt(test_pair_dir)

temp_dif_2=temp_dif_aug[:] # used to analysis slice of data
pair_need=test_pair[:,0].astype('int')
temp_dif_select=temp_dif_2[pair_need]
ind_predtemp_pair=np.hstack((test_sample_ind.reshape([-1,1]),pred_temp.reshape([-1,1])))

temp_pred=np.array([ind_predtemp_pair[np.where(ind_predtemp_pair[:,0]==temp_dif_select[i,0])[0],1] for i in range(temp_dif_select.shape[0])])
twin_temp_pred=np.array([ind_predtemp_pair[np.where(ind_predtemp_pair[:,0]==temp_dif_select[i,1])[0],1] for i in range(temp_dif_select.shape[0])])

temp_dif_select[:,6]=temp_pred.squeeze()
temp_dif_select[:,7]=twin_temp_pred.squeeze()
temp_dif_sort=temp_dif_select[np.argsort(-temp_dif_select[:,2],)]

select_position=temp_dif_sort[0,0].astype('int')
twin_position=temp_dif_sort[0,1].astype('int')

node_mag = np.array([10, 14, 20, 28, 40]).astype('int')
total_node=node_mag.sum()
los_divid=np.zeros([total_node])
length_interval=10.0/total_node
length_vec=np.arange(0,10,length_interval)
ave_temp_slct=np.ones_like(length_vec)*temp_dif_sort[0,3]
ave_temp_twin=np.ones_like(length_vec)*temp_dif_sort[0,4]

temp_divid_slct=np.zeros_like(los_divid)
temp_divid_twin=np.zeros_like(los_divid)
temp_select_pst=temp[select_position]
temp_twin_pst=temp[twin_position]

mole_divid_slct=np.zeros_like(los_divid)
mole_divid_twin=np.zeros_like(los_divid)
mole_select_pst=mole[select_position]
mole_twin_pst=mole[twin_position]

left_ind=0
right_ind=0
for i in range(node_mag.shape[0]):
    right_ind+=node_mag[i]
    temp_divid_slct[left_ind:right_ind]=temp_select_pst[i]
    temp_divid_twin[left_ind:right_ind] = temp_twin_pst[i]
    mole_divid_slct[left_ind:right_ind] = mole_select_pst[i]
    mole_divid_twin[left_ind:right_ind] = mole_twin_pst[i]
    left_ind=right_ind
#%% PLOT FIGURE this is used to generate fig 1 (b) of the paper, where spectra are same but distribution is different
plt.figure(figsize=(7,6))
plt.subplot(211)
waverange=np.arange(2375,2395.1,0.1)
plt.plot(waverange,spec[select_position,:],'b')
plt.plot(waverange,spec[twin_position,:],'r')
plt.xlabel("Wave number (cm$^{-1}$)")
plt.xlim([2375,2395])
plt.ylabel("Absorptivity")
plt.ylim([0,1])
plt.legend(labels=["Sample 1","Sample 2"],loc='upper right')
# plt.savefig('./figure_save/twins.jpg')

plt.subplot(223)
plt.plot(length_vec,temp_divid_slct,'b')
plt.plot(length_vec,temp_divid_twin,'r')
plt.plot(length_vec,ave_temp_slct,'b--')
plt.plot(length_vec,ave_temp_twin,'r--')
plt.xlabel("Light path (cm)")
plt.xlim([0,10])
plt.ylabel("Temperature (K)")
plt.ylim([600,2000])
plt.legend(labels=["Sample 1","Sample 2","Ave. Samp. 1","Ave. Samp. 2"],loc='lower right')

plt.subplot(224)
plt.plot(length_vec,mole_divid_slct,'b')
plt.plot(length_vec,mole_divid_twin,'r')
plt.xlabel("Light path (cm)")
plt.xlim([0,10])
plt.ylabel("Mole fraction")
plt.ylim([0.05,0.07])
plt.legend(labels=["Sample 1","Sample 2"],loc='lower right')
img_dir='figure_save/'+'twin_sample_'+str(select_position)+'_'+str(twin_position)+'jpg'
plt.tight_layout()
# plt.savefig(img_dir)
plt.show()
plt.close()
#%% uniform has totally different spectrum as nonuniform case
select_position=temp_dif_sort[0,0].astype('int')
# twin_position=temp_dif_sort[0,1].astype('int')

node_mag = np.array([10, 14, 20, 28, 40]).astype('int')
total_node=node_mag.sum()
los_divid=np.zeros([total_node])
length_interval=10.0/total_node
length_vec=np.arange(0,10,length_interval)

temp_divid_slct=np.zeros_like(los_divid)
temp_divid_twin=np.ones_like(los_divid)*1379.70
temp_select_pst=temp[select_position]
# temp_twin_pst=temp[twin_position]

mole_divid_slct=np.zeros_like(los_divid)
mole_divid_twin=np.ones_like(los_divid)*0.0652
mole_select_pst=mole[select_position]
# mole_twin_pst=mole[twin_position]

left_ind=0
right_ind=0
for i in range(node_mag.shape[0]):
    right_ind+=node_mag[i]
    temp_divid_slct[left_ind:right_ind]=temp_select_pst[i]
    # temp_divid_twin[left_ind:right_ind] = temp_twin_pst[i]
    mole_divid_slct[left_ind:right_ind] = mole_select_pst[i]
    # mole_divid_twin[left_ind:right_ind] = mole_twin_pst[i]
    left_ind=right_ind
#%% PLOT FIGURE his is used to generate fig 1 (a) of the paper, where spectra are different but average concentrations and temperatures are same
spec_uniform_dir='./input/uniform_case_for_twin8443/file/uniform_twin_8443_0.csv'
spec_uniform=np.loadtxt(spec_uniform_dir)
plt.figure(figsize=(7,6))
plt.subplot(211)
waverange=np.arange(2375,2395.1,0.1)
plt.plot(waverange,spec[select_position,:],'b')
plt.plot(waverange,spec_uniform,'r')
plt.xlabel("Wave number (cm$^{-1}$)")
plt.xlim([2375,2395])
plt.ylabel("Absorptivity")
plt.ylim([0,1])
plt.legend(labels=["Sample 1","Sample 2"],loc='upper right')
# plt.savefig('./figure_save/twins.jpg')

plt.subplot(223)
plt.plot(length_vec,temp_divid_slct,'b')
plt.plot(length_vec,temp_divid_twin,'r')
plt.xlabel("Light path (cm)")
plt.xlim([0,10])
plt.ylabel("Temperature (K)")
plt.ylim([600,2000])
plt.legend(labels=["Sample 1","Sample 2"],loc='lower right')

plt.subplot(224)
plt.plot(length_vec,mole_divid_slct,'b')
plt.plot(length_vec,mole_divid_twin,'r')
plt.xlabel("Light path (cm)")
plt.xlim([0,10])
plt.ylabel("Mole fraction")
plt.ylim([0.05,0.07])
plt.legend(labels=["Sample 1","Sample 2"],loc='lower right')
img_dir='figure_save/'+'uniform_vs_nonuniform_'+str(select_position)+'.jpg'
plt.tight_layout()
# plt.savefig(img_dir)
plt.show()
plt.close()

#%% plot figure 2 to demonstrate why we choose such a waveband
spec1_dir="./input/figure_input_600_2000/temp_600_2350_24300.csv"
spec2_dir="./input/figure_input_600_2000/temp_2000_2350_24300.csv"
spec1=np.loadtxt(spec1_dir)
spec2=np.loadtxt(spec2_dir)
waverange=np.arange(2350,2430.1,0.1)
plt.figure(2)
plt.subplot(211)
plt.plot(waverange,spec1,'b')
plt.xlabel("Wave number (cm$^{-1}$)")
plt.xlim([2350,2430])
plt.ylabel("Absorptivity")
plt.ylim([0,1])
plt.legend(labels=["X$_{CO2}$=0.07 T=600K"],loc='upper right')

plt.subplot(212)
plt.plot(waverange,spec2,'r')
plt.xlabel("Wave number (cm$^{-1}$)")
plt.xlim([2350,2430])
plt.ylabel("Absorptivity")
plt.ylim([0,1])
plt.legend(labels=["X$_{CO2}$=0.07 T=2000K"],loc='upper right')

img_dir='figure_save/'+'600_2000K'+'.jpg'
plt.tight_layout()
# plt.savefig(img_dir)
plt.show()
plt.close()

#%% plot figure 3 illness case

node_mag = np.array([10, 10, 10, 10, 10]).astype('int') # for illness case 1
# node_mag = np.array([10, 14, 20, 28, 40]).astype('int') # for illness case 2

total_node=node_mag.sum()
los_divid=np.zeros([total_node])
length_interval=10.0/total_node
length_vec=np.arange(0,10,length_interval)

# illness case 1
ill_1_dir='./input/illness_case/file/ill_1_samp10.csv'
ill_2_dir='./input/illness_case/file/ill_1_samp20.csv'
label1_dir='./input/illness_case/label/ill_1_samp10.csv'
label2_dir='./input/illness_case/label/ill_1_samp20.csv'
# # illness case 2
# ill_1_dir='./input/illness_case/file/ill_2_samp10.csv'
# ill_2_dir='./input/illness_case/file/ill_2_samp20.csv'
# label1_dir='./input/illness_case/label/ill_2_samp10.csv'
# label2_dir='./input/illness_case/label/ill_2_samp20.csv'


spec_1=np.loadtxt(ill_1_dir)
spec_2=np.loadtxt(ill_2_dir)
label_1=np.loadtxt(label1_dir)
label_2=np.loadtxt(label2_dir)


#%%
temp_divid_slct=np.zeros_like(los_divid)
temp_divid_twin=np.zeros_like(los_divid)
temp_select_pst=label_1[0]
temp_twin_pst=label_2[0]

mole_divid_slct=np.zeros_like(los_divid)
mole_divid_twin=np.zeros_like(los_divid)
mole_select_pst=label_1[1]
mole_twin_pst=label_2[1]

left_ind=0
right_ind=0
for i in range(node_mag.shape[0]):
    right_ind+=node_mag[i]
    temp_divid_slct[left_ind:right_ind]=temp_select_pst[i]
    temp_divid_twin[left_ind:right_ind] = temp_twin_pst[i]
    mole_divid_slct[left_ind:right_ind] = mole_select_pst[i]
    mole_divid_twin[left_ind:right_ind] = mole_twin_pst[i]
    left_ind=right_ind
#%% PLOT FIGURE
plt.figure(0)
plt.subplot(211)
waverange=np.arange(2375,2395.1,0.1)
plt.plot(waverange,spec_1,'b')
plt.plot(waverange,spec_2,'r')
plt.xlabel("Wave number (cm$^{-1}$)")
plt.xlim([2375,2395])
plt.ylabel("Absorptivity")
plt.ylim([0,1])
plt.legend(labels=["Sample 1","Sample 2"],loc='upper right')
# plt.savefig('./figure_save/twins.jpg')

plt.subplot(223)
plt.plot(length_vec,temp_divid_slct,'b')
plt.plot(length_vec,temp_divid_twin,'r')
plt.xlabel("Light path (cm)")
plt.xlim([0,10])
plt.ylabel("Temperature (K)")
plt.ylim([400,1200])
plt.legend(labels=["Sample 1","Sample 2"],loc='upper right')

plt.subplot(224)
plt.plot(length_vec,mole_divid_slct,'b')
plt.plot(length_vec,mole_divid_twin,'r')
plt.xlabel("Light path (cm)")
plt.xlim([0,10])
plt.ylabel("Mole fraction ")
plt.ylim([0.02,0.12])
plt.legend(labels=["Sample 1","Sample 2"],loc='upper right')
# img_dir='figure_save/'+'illness_case_2'+'jpg' # illness case 2
img_dir='figure_save/'+'illness_case_1'+'jpg' # illness case 1
plt.tight_layout()
plt.savefig(img_dir)
plt.show()
plt.close()

#%% figure 4 of the paper, the performance of models trained by uniform data on uniform data
vgg_dir="input/figure_uniform_pred_temp/temp_pred_vgg_b.csv"
xgb_dir="input/figure_uniform_pred_temp/predict_temp_uniform_xgboost.csv"
gpr_dir='input/figure_uniform_pred_temp/predict_temp_uniform_GPR.csv'
true_dir='input/figure_uniform_pred_temp/test_temp_uniform.csv'
gpr_pred=np.loadtxt(gpr_dir)
vgg_pred=np.loadtxt(vgg_dir)
xgb_pred=np.loadtxt(xgb_dir)
test_temp=np.loadtxt(true_dir)

#%%
start_point=300
select_section=40
section_point=np.arange(1,select_section+1,1)
plt.figure(4)
plt.subplot(311)
plt.plot(section_point,test_temp[start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,gpr_pred[start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Ground Truth","Prediction by GPR"],loc='upper right')

plt.subplot(312)
plt.plot(section_point,test_temp[start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,vgg_pred[start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Ground Truth","Prediction by VGG"],loc='upper right')

plt.subplot(313)
plt.plot(section_point,test_temp[start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,xgb_pred[start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Ground Truth","Prediction by BRF"],loc='upper right')

img_dir='figure_save/'+'uniform_prediction'+'jpg'
plt.tight_layout()
# plt.savefig(img_dir)
plt.show()
plt.close()
#%% figure 5 of the paper, the performance of uniform models on nonuniform data
vgg_dir="input/nonuniform_pred_spectra_twin/temp_dif_aug_1e-4_vgg.csv"
xgb_dir="input/nonuniform_pred_spectra_twin/temp_dif_aug_1e-4_xgb.csv"
gpr_dir='input/nonuniform_pred_spectra_twin/temp_dif_aug_1e-4_GPR.csv'
# true_dir='input/figure_uniform_pred_temp/test_temp_uniform.csv'
gpr_data=np.loadtxt(gpr_dir)
vgg_data=np.loadtxt(vgg_dir)
xgb_data=np.loadtxt(xgb_dir)

test_temp=xgb_data[:,3]
gpr_pred=gpr_data[:,6]
vgg_pred=vgg_data[:,6]
xgb_pred=xgb_data[:,6]
#%%
start_point=0
select_section=40
section_point=np.arange(1,select_section+1,1)
plt.figure(4)
plt.subplot(311)
plt.plot(section_point,test_temp[start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,gpr_pred[start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Ground Truth","Prediction by GPR"],loc='upper right')

plt.subplot(312)
plt.plot(section_point,test_temp[start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,vgg_pred[start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Ground Truth","Prediction by VGG"],loc='upper right')

plt.subplot(313)
plt.plot(section_point,test_temp[start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,xgb_pred[start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Ground Truth","Prediction by BRF"],loc='upper right')

img_dir='figure_save/'+'uniform_for_nonuniform_prediction_1e-4'+'jpg'
plt.tight_layout()
# plt.savefig(img_dir)
plt.show()
plt.close()

#%% figure 6 of the paper, the performance of uniform models on spectra twins
vgg_dir="input/nonuniform_pred_spectra_twin/temp_dif_aug_1e-5_vgg.csv"
xgb_dir="input/nonuniform_pred_spectra_twin/temp_dif_aug_1e-5_xgb.csv"
gpr_dir='input/nonuniform_pred_spectra_twin/temp_dif_aug_1e-5_GPR.csv'
# true_dir='input/figure_uniform_pred_temp/test_temp_uniform.csv'
gpr_data=np.loadtxt(gpr_dir)
vgg_data=np.loadtxt(vgg_dir)
xgb_data=np.loadtxt(xgb_dir)

criteria=1.e-5 # must change !!!
gpr_data=gpr_data[np.where(gpr_data[:,5]<=criteria)]
vgg_data=vgg_data[np.where(vgg_data[:,5]<=criteria)]
xgb_data=xgb_data[np.where(xgb_data[:,5]<=criteria)]

gpr_data=gpr_data[np.argsort(gpr_data[:,2])]
vgg_data=vgg_data[np.argsort(vgg_data[:,2])]
xgb_data=xgb_data[np.argsort(xgb_data[:,2])]

ref_temp=xgb_data[:,3]
twin_temp=xgb_data[:,4]

gpr_pred=gpr_data[:,6]
vgg_pred=vgg_data[:,6]
xgb_pred=xgb_data[:,6]

gpr_pred_twin=gpr_data[:,7]
vgg_pred_twin=vgg_data[:,7]
xgb_pred_twin=xgb_data[:,7]
#%%
start_point=0
select_section=40
section_point=np.arange(1,select_section+1,1)
plt.figure(4)
plt.subplot(311)
plt.plot(section_point,ref_temp[start_point:start_point+select_section],'b--',linewidth=0.5)
plt.plot(section_point,twin_temp[start_point:start_point+select_section],'r--',linewidth=0.5)
plt.plot(section_point,gpr_pred[start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,gpr_pred_twin[start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Reference","Twin","Ref. Prediction by GPR","Twin Prediction by GPR",],loc='upper right',
           fontsize=7)

plt.subplot(312)
plt.plot(section_point,ref_temp[start_point:start_point+select_section],'b--',linewidth=0.5)
plt.plot(section_point,twin_temp[start_point:start_point+select_section],'r--',linewidth=0.5)
plt.plot(section_point,vgg_pred[start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,vgg_pred_twin[start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Reference","Twin","Ref. Prediction by VGG","Twin Prediction by VGG",],loc='upper right',
           fontsize=7)


plt.subplot(313)
plt.plot(section_point,ref_temp[start_point:start_point+select_section],'b--',linewidth=0.5)
plt.plot(section_point,twin_temp[start_point:start_point+select_section],'r--',linewidth=0.5)
plt.plot(section_point,xgb_pred[start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,xgb_pred_twin[start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Reference","Twin","Ref. Prediction by BRF","Twin Prediction by BRF",],loc='upper right',
           fontsize=7)


img_dir='figure_save/'+'uniform_for_twin_1e-5_min'+'jpg'
plt.tight_layout()
# plt.savefig(img_dir)
plt.show()
plt.close()

#%% figure 8 of the paper, the performance of nonuniform spectra on spectra twin test set
import numpy as np
import matplotlib.pyplot as plt

temp_dif_aug=np.loadtxt('input/nonuniform_pred_spectra_twin/temp_dif_aug_1e-3_xgb.csv')
test_dir='./out_save/spectra_twin_split/nonuniform_test_sample_index_e3_new.csv'
test_sample_ind=np.loadtxt(test_dir).astype('int')

test_pair_dir='./out_save/spectra_twin_split/nonuniform_test_pair_e3_new.csv'
test_pair=np.loadtxt(test_pair_dir)

temp_dif_2=temp_dif_aug[:] # used to analysis slice of data
pair_need=test_pair[:,0].astype('int')
temp_dif_select=temp_dif_2[pair_need]
ref_temp=temp_dif_select[:,3]
twin_temp=temp_dif_select[:,4]
#%%
pred_temp_of_test_dir_vgg='./input/nonuniform_pred_temp/pred_temp_nonuniform_1e-3_new_VGG.csv'
pred_temp_vgg=np.loadtxt(pred_temp_of_test_dir_vgg)
if pred_temp_vgg.shape[0]!=test_sample_ind.shape[0]:
    print("size of predition index are not equal to size of prediction temp")

ind_predtemp_pair_vgg=np.hstack((test_sample_ind.reshape([-1,1]),pred_temp_vgg.reshape([-1,1])))
temp_pred_vgg=np.array([ind_predtemp_pair_vgg[np.where(ind_predtemp_pair_vgg[:,0]==temp_dif_select[i,0])[0],1] for i in range(temp_dif_select.shape[0])])
twin_temp_pred_vgg=np.array([ind_predtemp_pair_vgg[np.where(ind_predtemp_pair_vgg[:,0]==temp_dif_select[i,1])[0],1] for i in range(temp_dif_select.shape[0])])

pred_temp_of_test_dir_gpr='./input/nonuniform_pred_temp/pred_temp_nonuniform_1e-3_new_GPR.csv'
pred_temp_gpr=np.loadtxt(pred_temp_of_test_dir_gpr)
if pred_temp_gpr.shape[0]!=test_sample_ind.shape[0]:
    print("size of predition index are not equal to size of prediction temp")

ind_predtemp_pair_gpr=np.hstack((test_sample_ind.reshape([-1,1]),pred_temp_gpr.reshape([-1,1])))
temp_pred_gpr=np.array([ind_predtemp_pair_gpr[np.where(ind_predtemp_pair_gpr[:,0]==temp_dif_select[i,0])[0],1] for i in range(temp_dif_select.shape[0])])
twin_temp_pred_gpr=np.array([ind_predtemp_pair_gpr[np.where(ind_predtemp_pair_gpr[:,0]==temp_dif_select[i,1])[0],1] for i in range(temp_dif_select.shape[0])])

pred_temp_of_test_dir_xgb='./input/nonuniform_pred_temp/pred_temp_nonuniform_1e-3_new.csv'
pred_temp_xgb=np.loadtxt(pred_temp_of_test_dir_xgb)
if pred_temp_xgb.shape[0]!=test_sample_ind.shape[0]:
    print("size of predition index are not equal to size of prediction temp")

ind_predtemp_pair_xgb=np.hstack((test_sample_ind.reshape([-1,1]),pred_temp_xgb.reshape([-1,1])))
temp_pred_xgb=np.array([ind_predtemp_pair_xgb[np.where(ind_predtemp_pair_xgb[:,0]==temp_dif_select[i,0])[0],1] for i in range(temp_dif_select.shape[0])])
twin_temp_pred_xgb=np.array([ind_predtemp_pair_xgb[np.where(ind_predtemp_pair_xgb[:,0]==temp_dif_select[i,1])[0],1] for i in range(temp_dif_select.shape[0])])
#%%
start_point=0
select_section=40
section_point=np.arange(1,select_section+1,1)
plt.figure(8)
plt.subplot(311)
plt.plot(section_point,ref_temp[start_point:start_point+select_section],'b--',linewidth=0.5)
plt.plot(section_point,twin_temp[start_point:start_point+select_section],'r--',linewidth=0.5)
plt.plot(section_point,temp_pred_gpr[start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,twin_temp_pred_gpr[start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Reference","Twin","Ref. Prediction by GPR","Twin Prediction by GPR",],loc='upper right',
           fontsize=7)

plt.subplot(312)
plt.plot(section_point,ref_temp[start_point:start_point+select_section],'b--',linewidth=0.5)
plt.plot(section_point,twin_temp[start_point:start_point+select_section],'r--',linewidth=0.5)
plt.plot(section_point,temp_pred_vgg[start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,twin_temp_pred_vgg[start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Reference","Twin","Ref. Prediction by VGG","Twin Prediction by VGG",],loc='upper right',
           fontsize=7)


plt.subplot(313)
plt.plot(section_point,ref_temp[start_point:start_point+select_section],'b--',linewidth=0.5)
plt.plot(section_point,twin_temp[start_point:start_point+select_section],'r--',linewidth=0.5)
plt.plot(section_point,temp_pred_xgb[start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,twin_temp_pred_xgb[start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Reference","Twin","Ref. Prediction by BRF","Twin Prediction by BRF",],loc='upper right',
           fontsize=7)


img_dir='figure_save/'+'nonuniform_for_twin_1e-3_new'
plt.tight_layout()
# plt.savefig(img_dir)
plt.show()
plt.close()

#%% figure 10 of the paper generalization on more nonuniform profiles
import numpy as np
import matplotlib.pyplot as plt
from dataset import dataset_all_3

gpr_dir="input/10p_pred/gpr/"
gpr_pred, gpr_order=dataset_all_3(gpr_dir)

vgg_dir="input/10p_pred/vgg/"
vgg_pred, vgg_order=dataset_all_3(vgg_dir)

xgb_dir="input/10p_pred/xgb/"
xgb_pred, xgb_order=dataset_all_3(xgb_dir)

temp_truth_dir="input/data_10p/dens_temp_10p.csv"
temp_truth=np.loadtxt(temp_truth_dir)
#%%
start_point=0
select_section=40
section_point=np.arange(1,select_section+1,1)
plt.figure(10)
plt.subplot(311)
plt.plot(section_point,temp_truth[start_point:start_point+select_section],'b--',linewidth=0.5)
plt.plot(section_point,gpr_pred[0,start_point:start_point+select_section],'k',linewidth=0.5)
plt.plot(section_point,gpr_pred[1,start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,gpr_pred[2,start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Ground truth","GPR 1e-3","GPR 1e-4","GPR 1e-5",],loc='upper right',
           fontsize=7)

plt.subplot(312)
plt.plot(section_point,temp_truth[start_point:start_point+select_section],'b--',linewidth=0.5)
plt.plot(section_point,vgg_pred[0,start_point:start_point+select_section],'k',linewidth=0.5)
plt.plot(section_point,vgg_pred[1,start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,vgg_pred[2,start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Ground truth","VGG 1e-3","VGG 1e-4","VGG 1e-5",],loc='upper right',
           fontsize=7)

plt.subplot(313)
plt.plot(section_point,temp_truth[start_point:start_point+select_section],'b--',linewidth=0.5)
plt.plot(section_point,xgb_pred[0,start_point:start_point+select_section],'k',linewidth=0.5)
plt.plot(section_point,xgb_pred[1,start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,xgb_pred[2,start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Ground truth","BRF 1e-3","BRF 1e-4","BRF 1e-5",],loc='upper right',
           fontsize=7)


img_dir='figure_save/'+'10p_pred'+'jpg'
plt.tight_layout()
plt.savefig(img_dir)
plt.show()
plt.close()
#%% figure 11 generalization on more uniform profiles
import numpy as np
import matplotlib.pyplot as plt
from dataset import dataset_all_3, dataset_all

gpr_dir="input/uniform500_pred/gpr/"
gpr_pred, gpr_order=dataset_all_3(gpr_dir)

vgg_dir="input/uniform500_pred/vgg/"
vgg_pred, vgg_order=dataset_all_3(vgg_dir)

xgb_dir="input/uniform500_pred/xgb/"
xgb_pred, xgb_order=dataset_all_3(xgb_dir)

data_dir='./input/data_uniform/file'
label_dir='./input/data_uniform/label'
_,temp_truth,__,___=dataset_all(data_dir,label_dir)
#%%
start_point=0
select_section=40
section_point=np.arange(1,select_section+1,1)
plt.figure(10)
plt.subplot(311)
plt.plot(section_point,temp_truth[start_point:start_point+select_section],'b--',linewidth=0.5)
plt.plot(section_point,gpr_pred[0,start_point:start_point+select_section],'k',linewidth=0.5)
plt.plot(section_point,gpr_pred[1,start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,gpr_pred[2,start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Ground truth","GPR 1e-3","GPR 1e-4","GPR 1e-5",],loc='upper right',
           fontsize=7)

plt.subplot(312)
plt.plot(section_point,temp_truth[start_point:start_point+select_section],'b--',linewidth=0.5)
plt.plot(section_point,vgg_pred[0,start_point:start_point+select_section],'k',linewidth=0.5)
plt.plot(section_point,vgg_pred[1,start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,vgg_pred[2,start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Ground truth","VGG 1e-3","VGG 1e-4","VGG 1e-5",],loc='upper right',
           fontsize=7)

plt.subplot(313)
plt.plot(section_point,temp_truth[start_point:start_point+select_section],'b--',linewidth=0.5)
plt.plot(section_point,xgb_pred[0,start_point:start_point+select_section],'k',linewidth=0.5)
plt.plot(section_point,xgb_pred[1,start_point:start_point+select_section],'b',linewidth=0.5)
plt.plot(section_point,xgb_pred[2,start_point:start_point+select_section],'r',linewidth=0.5)
plt.xlabel("Sample index")
plt.xlim([0,40])
plt.ylabel("Temperature (K) ")
plt.ylim([500.,2000])
plt.legend(labels=["Ground truth","BRF 1e-3","BRF 1e-4","BRF 1e-5",],loc='upper right',
           fontsize=7)


img_dir='figure_save/'+'uniform500_pred'+'jpg'
plt.tight_layout()
# plt.savefig(img_dir)
plt.show()
plt.close()
#%% figure 9 of the paper, comparison between 10points and 5points
from dataset import *
import numpy as np
data_dir='./input/data_nonuniform/file'
label_dir='./input/data_nonuniform/label'
order_dir='./file_reading_order.csv'
spec,temp,mole,_=dataset_all_2(data_dir,label_dir,order_dir)
temp_dif_aug=np.loadtxt('input/nonuniform_pred_spectra_twin/temp_dif_aug_1e-3_xgb.csv')

data_dir='./input/data_10p/file'
label_dir='./input/data_10p/label'
# order_dir='file_reading_order.csv'
spec_10p,temp_10p,mole_10p, data_list=dataset_all(data_dir,label_dir)
temp_10p_dens=np.loadtxt("input/data_10p/dens_temp_10p.csv")

#%%
select_10p_pst=101
fontsize=8
temp_slc=temp_10p_dens[select_10p_pst]
temp_err=np.abs(temp_dif_aug[:,3]-temp_slc)
temp_err_min=temp_err.min()
corres_pst_nuf=np.where(temp_err==temp_err_min)[0].astype("int")

node_mag = np.array([10, 14, 20, 28, 40]).astype('int')
total_node=node_mag.sum()
los_divid=np.zeros([total_node])
length_interval=10.0/total_node
length_vec=np.arange(0,10,length_interval)

temp_divid_slct=np.zeros_like(los_divid)
temp_select_pst=temp[corres_pst_nuf].squeeze()

mole_divid_slct=np.zeros_like(los_divid)
mole_select_pst=mole[corres_pst_nuf].squeeze()
left_ind=0
right_ind=0
for i in range(node_mag.shape[0]):
    right_ind+=node_mag[i]
    temp_divid_slct[left_ind:right_ind]=temp_select_pst[i]
    mole_divid_slct[left_ind:right_ind] = mole_select_pst[i]
    left_ind=right_ind
ave_mole=np.ones_like(los_divid)*np.dot(mole_select_pst,node_mag)/total_node

node_mag_10p= np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]).astype('int')
total_node_10p=node_mag_10p.sum()
los_divid_10p=np.zeros([total_node_10p])
length_interval_10p=10.0/total_node_10p
length_vec_10p=np.arange(0,10,length_interval_10p)

temp_divid_slct_10p=np.zeros_like(los_divid_10p)
temp_select_pst_10p=temp_10p[select_10p_pst].squeeze()
ave_temp=np.ones_like(los_divid_10p)*temp_slc
mole_divid_slct_10p=np.zeros_like(los_divid_10p)
mole_select_pst_10p=mole_10p[select_10p_pst].squeeze()

left_ind=0
right_ind=0
for i in range(node_mag_10p.shape[0]):
    right_ind+=node_mag_10p[i]
    temp_divid_slct_10p[left_ind:right_ind]=temp_select_pst_10p[i]
    mole_divid_slct_10p[left_ind:right_ind] = mole_select_pst_10p[i]
    left_ind=right_ind
ave_mole_10p=np.ones_like(los_divid_10p)*np.dot(mole_select_pst_10p,node_mag_10p)/total_node_10p

plt.figure(11)
plt.subplot(211)
waverange=np.arange(2375,2395.1,0.1)
plt.plot(waverange,spec_10p[select_10p_pst,:].squeeze(),'b')
plt.plot(waverange,spec[corres_pst_nuf,:].squeeze(),'r')
plt.xlabel("Wave number (cm$^{-1}$)")
plt.xlim([2375,2395])
plt.ylabel("Absorptivity")
plt.ylim([0,1])
plt.legend(labels=["10 sections","5 sections"],loc='upper right',fontsize=fontsize)
# plt.savefig('./figure_save/twins.jpg')

plt.subplot(223)
plt.plot(length_vec_10p,temp_divid_slct_10p,'b')
plt.plot(length_vec,temp_divid_slct,'r')
plt.plot(length_vec_10p,ave_temp,'k--')
plt.xlabel("Light path (cm)")
plt.xlim([0,10])
plt.ylabel("Temperature (K)")
plt.ylim([600,2000])
plt.legend(labels=["10 sections","5 sections","Average"],loc='upper right',fontsize=fontsize)

plt.subplot(224)
plt.plot(length_vec_10p,mole_divid_slct_10p,'b')
plt.plot(length_vec,mole_divid_slct,'r')
plt.plot(length_vec_10p,ave_mole_10p,'b--')
plt.plot(length_vec,ave_mole,'r--')
plt.xlabel("Light path (cm)")
plt.xlim([0,10])
plt.ylabel("Mole fraction")
plt.ylim([0.05,0.07])
plt.legend(labels=["10 sections","5 sections","Ave. 10 sec.","Ave. 5 sec."],loc='upper right',fontsize=fontsize)
img_dir='figure_save/'+'10pvs5p_10-remade'
plt.tight_layout()
plt.savefig(img_dir)
plt.show()
plt.close()