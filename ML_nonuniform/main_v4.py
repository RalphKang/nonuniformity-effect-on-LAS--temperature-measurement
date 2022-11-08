# %%
from torch.utils.data import random_split, DataLoader, TensorDataset
import numpy as np
from torch.optim import AdamW, SGD
from train_vali_function import *
import torch.optim.lr_scheduler as lr_schedule
from sys import exit
from vgg import *
from dataset import dataset_all_2
from sklearn.metrics import mean_squared_error

"""
train vgg on datasets with different similarities
"""
def main():
    # def main(guide_lr=0.0, guide_wd=0.01, epoch_size=100, pretrain_mode=True, search_lr=False):
    sanity_check = False
    # set seed for numpy and pytorch
    torch.manual_seed(1)
    np.random.seed(1)

    # %% data_reading
    data_dir='./input/data_nonuniform/file'
    label_dir='./input/data_nonuniform/label'
    order_dir='./file_reading_order.csv'
    spec,temp,mole,_=dataset_all_2(data_dir,label_dir,order_dir)

    temp_dif_dir='./input/temp_dif_data/1e-5/temp_dens_comp.csv'
    temp_dif_select=np.loadtxt(temp_dif_dir)

    spec_max=np.loadtxt('input/spec_max.csv')
    spec_min=np.loadtxt('input/spec_min.csv')
    temp_bound=np.loadtxt('input/temp_bound.csv')

    train_index_dir='./input/nonuniform_train_sample_index_e5_new.csv'
    test_index_dir='./input/nonuniform_test_sample_index_e5_new.csv'
    train_index=np.loadtxt(train_index_dir)
    test_index=np.loadtxt(test_index_dir)
    # spec=np.log(spec)
    #%%
    temp_ave=temp_dif_select[:,3]
    #%% data_normalization
    spec_norm=(spec-spec_min)/(spec_max-spec_min)
    temp_norm=(temp_ave-temp_bound[0])/(temp_bound[1]-temp_bound[0])

    spec_norm=np.expand_dims(spec_norm,1)
    temp_norm=np.expand_dims(temp_norm,1)
    #%% split dataset
    train_ind_int=train_index.astype(int)
    test_ind_int=test_index.astype(int)

    train_spec=spec_norm[train_ind_int]
    test_spec=spec_norm[test_ind_int]
    spec_norm=np.expand_dims(spec_norm,1)

    train_temp=temp_norm[train_ind_int]
    test_temp=temp_norm[test_ind_int]

    # mole_max=mole.max(0)
    # mole_min=mole.min(0)
    # mole_norm=(mole-mole_min)/(mole_max-mole_min)
    # %% change to dataset
    train_tc = torch.from_numpy(train_spec)
    train_tcc = train_tc.float()
    label_train_tc = torch.from_numpy(train_temp)
    label_train_tcc = label_train_tc.float()
    data_set_all = TensorDataset(train_tcc, label_train_tcc)
    data_length = len(data_set_all)
    train_length = int(0.9 * data_length)
    vali_length = data_length - train_length
    train_set, vali_set = random_split(data_set_all, [train_length, vali_length])

    test_tc = torch.from_numpy(test_spec)
    test_tcc = test_tc.float()
    label_test_tc = torch.from_numpy(test_temp)
    label_test_tcc = label_test_tc.float()
    test_set = TensorDataset(test_tcc, label_test_tcc)
    # %% feed to dataloader( data generator)
    train_dl = DataLoader(train_set, batch_size=32, drop_last=True,shuffle=True)
    vali_dl = DataLoader(vali_set, batch_size=32, drop_last=True)
    test_dl = DataLoader(test_set, batch_size=32, drop_last=True)
    # %% **********************compile model**********************************************
    # choose model------------------
    model = VGG(make_layers(cfg['B'], batch_norm=False),1) # model A, final output 5 items
    model.to("cuda")

    # setting of loss,optimizer,lr_schedule and early stop----------------------------------
    model_save_dir = './model/vgg_nonuniform_1e-5.pt'
    performance_dir = './model/vgg_nonuniform_1e-5.txt'

    # lr setting----------------
    # search_lr = False  # find the best initial learning rate
    initial_lr = 5e-4  # no need to change for pretrain
    ## used to control the following set
    pretrain_mode=False
    guide_lr=5e-4 # the initial lr used
    search_lr=False
    epoch0 = 10  # warmup
    epoch_size=100

    if pretrain_mode:
        # model_path = './model/model_cnn_v6.pt'
        model.load_state_dict(torch.load(model_save_dir))
        history = np.loadtxt(performance_dir)
        initial_lr = history[-1, 0]  # find the latest lr rate
    if guide_lr > 0.0:
        initial_lr = guide_lr
    if search_lr:
        initial_lr = 1e-8
        pretrain_mode = False

    optimizer = AdamW(model.parameters(), lr=initial_lr)
    lr_scd = lr_schedule.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.6, patience=5, min_lr=1e-7,
                                           cooldown=2,
                                           threshold=0.0001)
    loss = nn.MSELoss(reduction="mean")
    # optimizer = SGD(model.parameters(),lr=initial_lr,momentum=0.9,nesterov=True)
    # training setting-----------

    epoch = epoch_size  # for official training

    if pretrain_mode:
        historical_best = history[-1, 2]
        # print("historical best obtained is {:.6f}".format(historical_best))
    else:
        historical_best = float('inf')
    run_time = 0
    tolerance = 20
    count_tole = 0
    train_loss_record = []
    vali_loss_record = []
    lr_record = []

    # writer = SummaryWriter('./run')
    # %% train and validation model
    # -------------warm up training-----------------------------------------
    if not pretrain_mode:
        for ep in range(epoch0):
            warm_up_lr(initial_lr=initial_lr, optimizer=optimizer, ite=ep, boundary=epoch0)
            train_loss_record = train_function(model=model, train_dl=train_dl, optimizer=optimizer, loss=loss, ep=ep,
                                               epoch=epoch0,
                                               train_loss_record=train_loss_record, lr_search=search_lr)
            if search_lr:
                exit()
            vali_loss_record = vali_function(model=model, model_save_dir=model_save_dir, vali_dl=vali_dl,
                                             loss=loss, ep=ep,
                                             epoch=epoch0, vali_loss_record=vali_loss_record,
                                             historical_best=historical_best)

            # vital parameters record
            old_lr = get_lr(optimizer)
            lr_record.append(old_lr)
            f = open(performance_dir, 'a')  # open file in append mode
            np.savetxt(f, np.c_[
                old_lr, train_loss_record[-1].cpu().detach().numpy(), vali_loss_record[-1].cpu().detach().numpy()])
            f.close()
    # --------------------official training----------------------------------------------------
    for ep in range(epoch):
        # change_lr(initial_lr,optimizer=optimizer,ite=ep,mode="exp",scale=30)
        # print("official training")
        train_loss_record = train_function(model=model, train_dl=train_dl, optimizer=optimizer, loss=loss, ep=ep,
                                           epoch=epoch, train_loss_record=train_loss_record, lr_search=False)
        vali_loss_record = vali_function(model=model, model_save_dir=model_save_dir, vali_dl=vali_dl,
                                         loss=loss, ep=ep, epoch=epoch, vali_loss_record=vali_loss_record,
                                         historical_best=historical_best)
        # learning rate on plateau change--
        old_lr = get_lr(optimizer)
        lr_scd.step(vali_loss_record[-1])
        new_lr = get_lr(optimizer)
        if old_lr != new_lr:
            model.load_state_dict(torch.load(model_save_dir))
        lr_record.append(old_lr)

        f = open(performance_dir, 'a')  # open file in append mode
        np.savetxt(f, np.c_[
            old_lr, train_loss_record[-1].cpu().detach().numpy(), vali_loss_record[-1].cpu().detach().numpy()])
        f.close()
        # # early stopping---------------------
        # if historical_best > vali_loss_record[-1]:
        #     historical_best = vali_loss_record[-1]
        #     count_tole = 0
        # if historical_best < vali_loss_record[-1]:
        #     count_tole += 1
        #     print("historical is better,historical_best={:.6f}, tolerance time={}".format(historical_best, count_tole))
        # if count_tole >= tolerance:
        #     print("best loss on validation set={:.6f}".format(historical_best))
        #     break
    #%%
    test_all=test_tcc.to("cuda")
    pred_test=model(test_all)
    pred_test_norm=pred_test.detach().cpu().numpy().squeeze()
    #%%
    pred_test=pred_test_norm*(temp_bound[1]-temp_bound[0])+temp_bound[0]
    test_temp_ori=test_temp*(temp_bound[1]-temp_bound[0])+temp_bound[0]
    mse_test=mean_squared_error(test_temp_ori,pred_test)

    np.savetxt('output/pred_temp_nonuniform_1e-5_new_VGG.csv',pred_test)
if __name__ == '__main__':
    main()