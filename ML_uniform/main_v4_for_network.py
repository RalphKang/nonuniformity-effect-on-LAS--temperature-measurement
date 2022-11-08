# %%
from torch.utils.data import random_split, DataLoader, TensorDataset
import numpy as np
from torch.optim import AdamW, SGD
from train_vali_function import *
import torch.optim.lr_scheduler as lr_schedule
# from torch.utils.tensorboard import SummaryWriter
from sys import exit
from network_archive.vgg import *
from dataset import dataset_all, all_statistic
from sklearn.metrics import mean_squared_error


def main():
    sanity_check = False
    # set seed for numpy and pytorch
    torch.manual_seed(1)
    np.random.seed(1)

    # %% data_reading
    data_dir = './input/data_uniform/file'
    label_dir = './input/data_uniform/label'
    spec, temp, _, __ = dataset_all(data_dir, label_dir)

    # %% data_normalization
    data_size = temp.shape
    data_length = data_size[0]
    train_length = int(data_length * 0.9)
    train_spec = spec[:train_length]

    spec_max = train_spec.max(0)
    spec_min = train_spec.min(0)

    temp_max = temp.max(0)
    temp_min = temp.min(0)
    temp_norm = (temp - temp_min) / (temp_max - temp_min)

    spec_norm = (spec - spec_min) / (spec_max - spec_min)

    spec_norm = np.expand_dims(spec_norm, 1)
    temp_norm = np.expand_dims(temp_norm, 1)
    # %% split dataset

    train_spec = spec_norm[:train_length]
    test_spec = spec_norm[train_length:]
    spec_norm = np.expand_dims(spec_norm, 1)

    train_temp = temp_norm[:train_length]
    test_temp = temp_norm[train_length:]

    # mole_max=mole.max(0)
    # mole_min=mole.min(0)
    # mole_norm=(mole-mole_min)/(mole_max-mole_min)
    # %% change to dataset
    # split train dataset into train and validation
    train_tc = torch.from_numpy(train_spec)
    train_tcc = train_tc.float()
    label_train_tc = torch.from_numpy(train_temp)
    label_train_tcc = label_train_tc.float()
    data_set_all = TensorDataset(train_tcc, label_train_tcc)
    data_length = len(data_set_all)
    train_length = int(0.9 * data_length)
    vali_length = data_length - train_length
    train_set, vali_set = random_split(data_set_all, [train_length, vali_length])
    # transform numpy to torch tensor
    test_tc = torch.from_numpy(test_spec)
    test_tcc = test_tc.float()
    label_test_tc = torch.from_numpy(test_temp)
    label_test_tcc = label_test_tc.float()
    test_set = TensorDataset(test_tcc, label_test_tcc)
    # %% feed to dataloader( data generator)
    train_dl = DataLoader(train_set, batch_size=32, drop_last=True, shuffle=True)
    vali_dl = DataLoader(vali_set, batch_size=32, drop_last=True)
    test_dl = DataLoader(test_set, batch_size=32, drop_last=True)
    # %% **********************compile model**********************************************
    # choose model------------------
    model = VGG(make_layers(cfg['A'], batch_norm=False), 1)  # model A, final output 5 items
    model.to("cuda")

    # setting of loss,optimizer,lr_schedule and early stop----------------------------------
    model_save_dir = './model/vgg_A_uniform_test.pt'
    performance_dir = './model/vgg_A_uniform_test.txt'

    # lr setting----------------
    # search_lr = False  # find the best initial learning rate
    initial_lr = 5e-4  # no need to change for pretrain
    ## used to control the following set
    pretrain_mode = False # whether has a pretrian model to load
    guide_lr = 5.e-4  # the initial lr used
    search_lr = False
    epoch_size = 100

    if pretrain_mode:
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
    epoch0 = 10  # warmup
    epoch = epoch_size  # for official training

    if pretrain_mode:
        historical_best = history[-1, 2]
        print("historical best obtained is {:.6f}".format(historical_best))
    else:
        historical_best = float('inf')
    run_time = 0
    tolerance = 20
    count_tole = 0
    train_loss_record = []
    vali_loss_record = []
    lr_record = []

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
            # writer.add_scalars('loss', {"train": train_loss_record[-1], "vali": vali_loss_record[-1]},
            #                    ep)  # tensorboard visualize
            # writer.close()
            # performance_record = np.c_[lr_record, train_loss_record, vali_loss_record]
            f = open(performance_dir, 'a')  # open file in append mode
            np.savetxt(f, np.c_[
                old_lr, train_loss_record[-1].cpu().detach().numpy(), vali_loss_record[-1].cpu().detach().numpy()])
            f.close()
    # --------------------official training----------------------------------------------------
    for ep in range(epoch):
        # change_lr(initial_lr,optimizer=optimizer,ite=ep,mode="exp",scale=30)
        print("official training")
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
    # %% test and check result
    test_all = test_tcc.to("cuda")
    temp_test_pred = model(test_all)
    pred_test_norm = temp_test_pred.detach().cpu().numpy().squeeze()
    # %%
    train_all = train_tcc[:train_length].to("cuda")
    temp_train_pred = model(train_all)
    pred_train_norm = temp_train_pred.detach().cpu().numpy().squeeze()
    # %%
    temp_train_orig = train_temp[:train_length] * (temp_max - temp_min) + temp_min
    temp_train_pred_orig = pred_train_norm * (temp_max - temp_min) + temp_min
    temp_test_orig = test_temp * (temp_max - temp_min) + temp_min
    temp_test_pred_orig = pred_test_norm * (temp_max - temp_min) + temp_min
    # %%
    metric_test = all_statistic(temp_test_orig, temp_test_pred_orig)
    metric_train = all_statistic(temp_train_orig, temp_train_pred_orig)
    metric_train_test = np.vstack((metric_train, metric_test))
    # %%
    metric_dir = 'output/vgg_A_metric.csv'
    np.savetxt(metric_dir, metric_train_test)


if __name__ == '__main__':
    main()
