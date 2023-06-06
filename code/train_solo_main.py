import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import argparse
import time
import copy
import gc
import importlib
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from monai.transforms import (
    Compose,
    AsDiscrete,
)
import json
from utils.metric import HausdorffDistanceMetric
from monai.metrics import DiceMetric
import time
import copy
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

from net.official_nnunet.generic_modular_UNet import *
from utils.utils import *

def train(cfg, model, metric_function, post_pred, data_loader, optimizer, scheduler, loss_fun, device, client_idx):

    dice_metric, hausdorff_metric = metric_function
    model = DDP(model.cuda(device), device_ids=[device], output_device=device, find_unused_parameters=True)

    model.train()
    scaler = GradScaler(init_scale=2.**8)

    progress_bar = tqdm(range(len(data_loader)))
    tr_it = iter(data_loader)

    loss_all = 0
    dataset_size = 0
    running_loss = 0.0
    len_data_loader = len(data_loader)
    dice, hausdorff = [], []

    for itr in progress_bar:
        batch = next(tr_it)
        optimizer.zero_grad()

        data, target, data_name = (
            batch["image"],
            batch["mask"],
            batch["name"][0],
        )

        # background = 1-foreground
        with torch.no_grad():
            if client_idx == 0:
                target_new = torch.index_select(target, 1, torch.tensor([0, 1]).to(target.device))
                target_new[:,0] = 1 - target_new[:,1]
            elif client_idx == 1:
                target_new = torch.index_select(target, 1, torch.tensor([0, 3, 4]).to(target.device))
                target_new[:,0] = 1 - (target_new[:,1] + target_new[:,2])
            elif client_idx == 2:
                target_new = torch.index_select(target, 1, torch.tensor([0, 2]).to(target.device))
                target_new[:,0] = 1 - target_new[:,1]
            del target
            target = target_new
            del target_new

        data, target = data.cuda(device), target.cuda(device)
            
        with autocast(enabled=True):
            
            output = model(data)    
                
            loss = loss_fun(output, target)

            dist.barrier()  # synchronizes all processes
            dist.all_reduce(loss, op=torch.distributed.ReduceOp.AVG,)  # get mean loss for all processes

            loss_all = loss_all + loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(optimizer)
            scaler.update()
            
            # calculate mertic
            outputs = []
            for j in output[:,4]:
                outputs.append(post_pred(j))
            outputs = torch.tensor(torch.stack(outputs), dtype=torch.uint8)
            # metric (n*d, c, h, w)
            outputs = outputs.permute([0, 2, 1, 3, 4]).flatten(0, 1)
            masks = target.permute([0, 2, 1, 3, 4]).flatten(0, 1)

            for j, i in zip(range(cfg.task_num), range(1,cfg.task_num+1)):
                hausdorff_metric[j](y_pred=outputs[:,i].unsqueeze(1), y=masks[:,i].unsqueeze(1))
                dice_metric[j](y_pred=outputs[:,i].unsqueeze(1), y=masks[:,i].unsqueeze(1))

        running_loss = running_loss + (loss.item() * cfg.batch_size)
        dataset_size = dataset_size + cfg.batch_size
        
        scheduler.step()
    
    dist.barrier()
    for i in range(cfg.task_num):
        dice.append(dice_metric[i].aggregate().item())
        hausdorff.append(hausdorff_metric[i].aggregate().item())
        dice_metric[i].reset()
        hausdorff_metric[i].reset()
        
    loss = loss_all / len_data_loader

    model.to('cpu')

    del data, target, loss_all, loss_fun, output, masks, outputs, running_loss
    gc.collect()
    torch.cuda.empty_cache()

    return loss, dice, hausdorff


def test(cfg, model, metric_function, post_pred, data_loader, loss_fun, device, client_idx):

    dice_metric, hausdorff_metric = metric_function

    model = DDP(model.cuda(device), device_ids=[device], output_device=device, find_unused_parameters=True)

    torch.set_grad_enabled(False)
    progress_bar = tqdm(range(len(data_loader)))
    val_it = iter(data_loader)
    dice, hausdorff = [], []

    with torch.no_grad():
        for itr in progress_bar:
            model.eval()
            batch = next(val_it)

            data, target = (
                batch["image"],
                batch["mask"],
            )

            if client_idx == 0:
                target_new = torch.index_select(target, 1, torch.tensor([0, 1]).to(target.device))
                target_new[:,0] = 1 - target_new[:,1]
            elif client_idx == 1:
                target_new = torch.index_select(target, 1, torch.tensor([0, 3, 4]).to(target.device))
                target_new[:,0] = 1 - (target_new[:,1] + target_new[:,2])
            elif client_idx == 2:
                target_new = torch.index_select(target, 1, torch.tensor([0, 2]).to(target.device))
                target_new[:,0] = 1 - target_new[:,1]
            del target
            target = target_new
            del target_new

            data, target = data.cuda(device), target.cuda(device)

            output = sliding_window_inference(data, cfg.roi_size, cfg.sw_batch_size, model, 0.5).cpu()

            outputs = []
            for j in output:
                outputs.append(post_pred(j))
            outputs = torch.tensor(torch.stack(outputs), dtype=torch.uint8)

            outputs = outputs.permute([0, 2, 1, 3, 4]).flatten(0, 1)
            masks = target.permute([0, 2, 1, 3, 4]).flatten(0, 1).cpu()
 
            for j, i in zip(range(cfg.task_num), range(1,cfg.task_num+1)):
                hausdorff_metric[j](y_pred=outputs[:,i].unsqueeze(1), y=masks[:,i].unsqueeze(1))
                dice_metric[j](y_pred=outputs[:,i].unsqueeze(1), y=masks[:,i].unsqueeze(1))

    dist.barrier()
    for i in range(cfg.task_num):
        dice.append(dice_metric[i].aggregate().item())
        hausdorff.append(hausdorff_metric[i].aggregate().item())
        dice_metric[i].reset()
        hausdorff_metric[i].reset()

    model.to('cpu')

    del data, target, output, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return dice, hausdorff

def initialize(cfg):
    train_loaders, val_loaders, test_loaders = [], [], []
    trainsets, valsets, testsets = [], [], []
    dice_metric, hausdorff_metric = [], []

    site = cfg.client_name
    if 'client_1' == site:
        client_idx = 0
        cfg.task_num = 1
    elif 'client_2' == site:
        client_idx = 1
        cfg.task_num = 2
    elif 'client_3' == site:
        client_idx = 2
        cfg.task_num = 1

    for i in range(cfg.task_num):
        dice_metric.append(DiceMetric(reduction="mean"))
        hausdorff_metric.append(HausdorffDistanceMetric(reduction="mean"))
    metric_function = [dice_metric, hausdorff_metric]

    post_pred = Compose([
        AsDiscrete(argmax=True, to_onehot=cfg.task_num+1),
    ])

    # data sequence
    cfg.data_json_dir = cfg.data_dir + "client_data/path_json/"

    cfg.client_num = 1

    conv_op_kernel_sizes = ((3, 3, 3),
                            (3, 3, 3),
                            (3, 3, 3),
                            (3, 3, 3),
                            (3, 3, 3),
                            (3, 3, 3))
    pool_op_kernel_sizes = ((1, 1, 1),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2),
                            (1, 2, 2))
    base_num_features = cfg.base_num_features
    input_modalities = 1 
    blocks_per_stage_encoder = (2, 2, 2, 2, 2, 2)
    blocks_per_stage_decoder = (2, 2, 2, 2, 2)
    feat_map_mult_on_downscale = 2
    num_classes = cfg.task_num+1 # out_channel
    max_features = 320
    model = PlainConvUNet(input_modalities, base_num_features, blocks_per_stage_encoder, feat_map_mult_on_downscale,
            pool_op_kernel_sizes, conv_op_kernel_sizes, get_default_network_config(3, dropout_p=None),
            num_classes, blocks_per_stage_decoder, deep_supervision=True, upscale_logits=True, max_features=max_features, 
            client_num=1, cfg=cfg)

    init_weights(model, init_type='kaiming')

    loss_fun = DiceBceMultilabelLoss(cfg)

    client_num_list = []

    with open(cfg.data_json_dir + site + '.json', "r") as f:
        cfg.data_json = json.load(f)

    trainset = get_train_dataset(cfg)
    valset = get_val_dataset(cfg)
    testset = get_test_dataset(cfg)

    print(f'[Client {site[-1]}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}') if cfg.local_rank == 0 else 0
    trainsets.append(trainset)
    valsets.append(valset)
    testsets.append(testset)
    client_num_list.append(len(trainset))

    for idx in range(len(trainsets)):
        trainset = trainsets[idx]
        valset = valsets[idx]
        testset = testsets[idx]

        train_loaders.append(get_train_dataloader(trainset, cfg))
        val_loaders.append(get_val_dataloader(valset, cfg))
        test_loaders.append(get_val_dataloader(testset, cfg))

    return model, loss_fun, metric_function, post_pred, site, trainsets, testsets, train_loaders, val_loaders, test_loaders, client_idx

def main(device, cfg):
    cfg.local_rank = device

    torch.cuda.set_device(device)                      
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method=cfg.dist_url,
    	world_size=cfg.world_size,
    	rank=cfg.local_rank,
    )

    torch.manual_seed(0)

    cfg.save_path = '{}/seed{}'.format(cfg.output_dir, cfg.seed)
    exp_folder = cfg.exp_folder

    cfg.save_path = os.path.join(cfg.save_path, exp_folder)
    if not os.path.exists(cfg.save_path) and cfg.local_rank == 0:
        os.makedirs(cfg.save_path)

    print(cfg) if cfg.local_rank == 0 else 0

    cfg.device = device

    server_model, loss_fun, metric_function, post_pred, site, train_datasets, _, train_loaders, val_loaders, test_loaders, client_idx  = initialize(cfg)

    print('# Deive:', cfg.multi_gpu) if cfg.local_rank == 0 else 0
    print('# Training Clients:{}'.format(site)) if cfg.local_rank == 0 else 0

    log = cfg.log
    if log and cfg.local_rank == 0:
        log_path = cfg.save_path + 'log'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'FL.log'), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        for k in list(vars(cfg).keys()):
            logfile.write('{}: {}\n'.format(k, vars(cfg)[k]))

    client_num = 1
    models = [copy.deepcopy(server_model) for _ in range(client_num)]
    total_steps = [len(train_datasets[idx]) for idx in range(client_num)]
    optimizers = [get_optimizer(models[idx], cfg) for idx in range(client_num)]
    after_scheduler = [get_scheduler(cfg, optimizers[idx], total_steps[idx]) for idx in range(client_num)]
    scheduler = [GradualWarmupScheduler(cfg, optimizers[idx], multiplier=cfg.multiplier, total_epoch=total_steps[idx]*cfg.warmup_epoch, after_scheduler=after_scheduler[idx]) for idx in range(client_num)]

    # resume training from last checkpoint
    if cfg.resume:
        checkpoint = torch.load(cfg.weights_path)
        if isinstance(checkpoint,torch.nn.DataParallel):
            checkpoint = checkpoint.module
        models[0].load_state_dict(checkpoint[f'Client_model'])

        if 'optim_0' in list(checkpoint.keys()):
            optimizers[0].load_state_dict(checkpoint[f'optim_0'])
            scheduler[0].load_state_dict(checkpoint[f'scheduler_0'])

        best_epoch, best_dice, best_hd  = checkpoint['best_epoch'], checkpoint['best_dice'], checkpoint['best_hd']
        start_iter = int(checkpoint['a_iter']) + 1
        print(f'Last time best:{best_epoch} dice :{best_dice}, HD :{best_hd}') if cfg.local_rank == 0 else 0
        print('Resume training from epoch {}'.format(start_iter)) if cfg.local_rank == 0 else 0
    else:
        best_epoch = 0
        best_dice = [0. for _ in range(client_num)]
        best_hd = [1000. for _ in range(client_num)]
        start_iter = 0

    # start training
    best_changed = False
    for a_iter in range(start_iter, cfg.epochs):
        print("============ Train epoch {} ============".format(a_iter))  if cfg.local_rank == 0 else 0
        if log and cfg.local_rank == 0:
            logfile.write("============ Train epoch {} ============\n".format(a_iter))
        for model in models:
            train_loaders[0].sampler.set_epoch(a_iter)
            for _ in range(cfg.local_round):
                train_loss, train_dice, train_hausdorff = train(cfg, model, metric_function, post_pred, train_loaders[0], optimizers[0], scheduler[0], loss_fun, device, client_idx)
                mean_dice, mean_hd = np.mean(train_dice), np.mean(train_hausdorff)
                if client_idx == 0:
                    print(' Site-{:<10s}| Train Loss: {:.4f} \n \
                        Train Dice_Kidney: {:.4f} \n \
                        Train HD_Kidney: {:.4f} '.format \
                        (site, train_loss, \
                            train_dice[0], \
                            train_hausdorff[0]))  if cfg.local_rank == 0 else 0
                    if log and cfg.local_rank == 0:
                        logfile.write(' Site-{:<10s}| Train Loss: {:.4f} \n \
                            Train Dice_Kidney: {:.4f} \n \
                            Train HD_Kidney: {:.4f} '.format \
                            (site, train_loss, \
                                train_dice[0],\
                                train_hausdorff[0]))
                elif client_idx == 1:
                    print(' Site-{:<10s}| Train Loss: {:.4f} \n \
                            Train Dice_Spleen: {:.4f}, Train Dice_Pancreas: {:.4f} \n \
                            Train HD_Spleen: {:.4f}, Train HD_Pancreas: {:.4f} '.format \
                            (site, train_loss, \
                                train_dice[0], train_dice[1],\
                                train_hausdorff[0], train_hausdorff[1]))  if cfg.local_rank == 0 else 0
                    if log and cfg.local_rank == 0:
                        logfile.write(' Site-{:<10s}| Train Loss: {:.4f} \n \
                            Train Dice_Spleen: {:.4f}, Train Dice_Pancreas: {:.4f} \n \
                            Train HD_Spleen: {:.4f}, Train HD_Pancreas: {:.4f} '.format \
                            (site, train_loss, \
                                train_dice[0], train_dice[1],\
                                train_hausdorff[0], train_hausdorff[1]))
                elif client_idx == 2:
                    print(' Site-{:<10s}| Train Loss: {:.4f} \n \
                        Train Dice_Liver: {:.4f} \n \
                        Train HD_Liver: {:.4f} '.format \
                        (site, train_loss, \
                            train_dice[0], \
                            train_hausdorff[0]))  if cfg.local_rank == 0 else 0
                    if log and cfg.local_rank == 0:
                        logfile.write(' Site-{:<10s}| Train Loss: {:.4f} \n \
                            Train Dice_Liver: {:.4f} \n \
                            Train HD_Liver: {:.4f} '.format \
                            (site, train_loss, \
                                train_dice[0],\
                                train_hausdorff[0]))
        torch.cuda.empty_cache()

        with torch.no_grad():
            dist.barrier()
            # val
            if a_iter >= cfg.start_eval_epoch and np.mod(a_iter, cfg.eval_epochs) == 0:
                val_dice_list = [None for j in range(client_num)]
                val_hd_list = [None for j in range(client_num)]
                print('============== {} =============='.format('Global Validation'))  if cfg.local_rank == 0 else 0
                if log and cfg.local_rank == 0:
                    logfile.write('============== {} ==============\n'.format('Global Validation'))
                for _, model in enumerate(models):
                    val_loaders[0].sampler.set_epoch(a_iter)
                    val_dice, val_hausdorff = test(cfg, model, metric_function, post_pred, val_loaders[0], loss_fun, device, client_idx)
                    mean_dice, mean_hd = np.mean(val_dice), np.mean(val_hausdorff)
                    val_dice_list[0] = mean_dice
                    val_hd_list[0] = mean_hd
                    if client_idx == 0:
                        print(' Site-{:<10s} \n \
                                Val Dice_Kidney: {:.4f} \n \
                                Val HD_Kidney: {:.4f} \n '.format \
                                (site, \
                                val_dice[0], \
                                val_hausdorff[0]))  if cfg.local_rank == 0 else 0
                        if log and cfg.local_rank == 0:
                            logfile.write(' Site-{:<10s} \n \
                                Val Dice_Kidney: {:.4f} \n \
                                Val HD_Kidney: {:.4f} \n'.format \
                                (site, \
                                val_dice[0], \
                                val_hausdorff[0]))
                    elif client_idx == 1:
                        print(' Site-{:<10s} \n \
                                Val Dice_Spleen: {:.4f}, Val Dice_Pancreas: {:.4f}\n \
                                Val HD_Spleen: {:.4f}, Val HD_Pancreas: {:.4f}\n '.format \
                                (site, \
                                val_dice[0], val_dice[1], \
                                val_hausdorff[0], val_hausdorff[1]))  if cfg.local_rank == 0 else 0
                        if log and cfg.local_rank == 0:
                            logfile.write(' Site-{:<10s} \n \
                                Val Dice_Spleen: {:.4f}, Val Dice_Pancreas: {:.4f}\n \
                                Val HD_Spleen: {:.4f}, Val HD_Pancreas: {:.4f}\n'.format \
                                (site, \
                                val_dice[0],val_dice[1], \
                                val_hausdorff[0], val_hausdorff[1]))
                    elif client_idx == 2:
                        print(' Site-{:<10s} \n \
                                Val Dice_Liver: {:.4f} \n \
                                Val HD_Liver: {:.4f} \n '.format \
                                (site, \
                                val_dice[0], \
                                val_hausdorff[0]))  if cfg.local_rank == 0 else 0
                        if log and cfg.local_rank == 0:
                            logfile.write(' Site-{:<10s} \n \
                                Val Dice_Liver: {:.4f} \n \
                                Val HD_Liver: {:.4f} \n'.format \
                                (site, \
                                val_dice[0], \
                                val_hausdorff[0]))

                torch.cuda.empty_cache()

                # test
                test_dice_list = [None for j in range(client_num)]
                test_hd_list = [None for j in range(client_num)]
                print('============== {} =============='.format('Test'))  if cfg.local_rank == 0 else 0
                if log and cfg.local_rank == 0:
                    logfile.write('============== {} ==============\n'.format('Test'))
                for _, model in enumerate(models):
                    test_loaders[0].sampler.set_epoch(a_iter)
                    test_dice, test_hausdorff = test(cfg, model, metric_function, post_pred, test_loaders[0], loss_fun, device, client_idx)
                    mean_dice, mean_hd = np.mean(test_dice), np.mean(test_hausdorff)
                    test_dice_list[0] = mean_dice
                    test_hd_list[0] = mean_hd

                    if client_idx == 0:
                        print(' Test site-{:<10s}| Epoch:{} \n \
                                Test Dice_Kidney: {:.4f} \n \
                                Test HD_Kidney: {:.4f} '.format \
                                (site, a_iter, \
                                test_dice[0],\
                                test_hausdorff[0]))  if cfg.local_rank == 0 else 0
                        if log and cfg.local_rank == 0:
                            logfile.write(' Test site-{:<10s}| Epoch:{} \n \
                                Test Dice_Kidney: {:.4f} \n \
                                Test HD_Kidney: {:.4f} '.format \
                                (site, a_iter, \
                                test_dice[0],\
                                test_hausdorff[0]))
                            logfile.flush()
                    elif client_idx == 1:
                        print(' Test site-{:<10s}| Epoch:{} \n \
                                Test Dice_Spleen: {:.4f}, Test Dice_Pancreas: {:.4f}\n \
                                Test HD_Spleen: {:.4f}, Test HD_Pancreas: {:.4f}'.format \
                                (site, a_iter, \
                                test_dice[0], test_dice[1],\
                                test_hausdorff[0], test_hausdorff[1]))  if cfg.local_rank == 0 else 0
                        if log and cfg.local_rank == 0:
                            logfile.write(' Test site-{:<10s}| Epoch:{} \n \
                                Test Dice_Spleen: {:.4f}, Test Dice_Pancreas: {:.4f}\n \
                                Test HD_Spleen: {:.4f}, Test HD_Pancreas: {:.4f} '.format \
                                (site, a_iter, \
                                test_dice[0],test_dice[1],\
                                test_hausdorff[0], test_hausdorff[1]))
                            logfile.flush()
                    elif client_idx == 2:
                        print(' Test site-{:<10s}| Epoch:{} \n \
                                Test Dice_Liver: {:.4f} \n \
                                Test HD_Liver: {:.4f} '.format \
                                (site, a_iter, \
                                test_dice[0],\
                                test_hausdorff[0]))  if cfg.local_rank == 0 else 0
                        if log and cfg.local_rank == 0:
                            logfile.write(' Test site-{:<10s}| Epoch:{} \n \
                                Test Dice_Liver: {:.4f} \n \
                                Test HD_Liver: {:.4f} '.format \
                                (site, a_iter, \
                                test_dice[0],\
                                test_hausdorff[0]))
                            logfile.flush()

                torch.cuda.empty_cache()

                # record the best dice
                if val_dice_list[0] > best_dice[0]:
                    best_dice[0] = val_dice_list[0]
                    best_hd[0] = val_hd_list[0]
                    best_epoch = a_iter
                    best_changed=True
                    if cfg.local_rank == 0:
                        print(' Best Epoch:{}. Best Val Dice:{:.4f}. Best Val HD:{:.4f}. Test Dice:{:.4f}. Test HD:{:.4f}'.format(best_epoch, np.mean(val_dice_list), np.mean(val_hd_list), np.mean(test_dice_list), np.mean(test_hd_list)))
                    if log and cfg.local_rank == 0:
                        logfile.write(' Best Epoch:{}. Best Val Dice:{:.4f}. Best Val HD:{:.4f}. Test Dice:{:.4f}. Test HD:{:.4f}\n'.format(best_epoch, np.mean(val_dice_list), np.mean(val_hd_list), np.mean(test_dice_list), np.mean(test_hd_list)))
                        logfile.flush()

                # save model weights and information for the best dice
                if best_changed:
                    print(' Saving the local and server checkpoint to {}...'.format(cfg.save_path))  if cfg.local_rank == 0 else 0
                    if log and cfg.local_rank == 0: 
                        logfile.write(' Saving the local and server checkpoint to {}...\n'.format(cfg.save_path))
                
                    model_dicts = { 'Client_model': models[0].state_dict(),
                                    'best_epoch': best_epoch,
                                    'best_dice': best_dice,
                                    'best_hd': best_hd,
                                    'a_iter': a_iter}
                    
                    model_dicts['optim_0'] = optimizers[0].state_dict()
                    model_dicts['scheduler_0'] = scheduler[0].state_dict()
                    if dist.get_rank() == 0:
                        torch.save(model_dicts, cfg.save_path + '/best_weights.pth')
                        torch.save(model_dicts, cfg.save_path + '/latest_weights.pth')
                    best_changed = False
                else:
                    # save the latest model weights and other information
                    print(' Saving the latest checkpoint to {}...'.format(cfg.save_path))  if cfg.local_rank == 0 else 0
                    if log and cfg.local_rank == 0:
                        logfile.write(' Saving the latest checkpoint to {}...\n'.format(cfg.save_path))
                    
                    model_dicts = { 'Client_model': models[0].state_dict(),
                                    'best_epoch': best_epoch,
                                    'best_dice': best_dice,
                                    'best_hd': best_hd,
                                    'a_iter': a_iter}
                    model_dicts['optim_0'] = optimizers[0].state_dict()
                    model_dicts['scheduler_0'] = scheduler[0].state_dict()
                    if dist.get_rank() == 0:
                        torch.save(model_dicts, cfg.save_path + '/latest_weights.pth')
                del model_dicts
                gc.collect()
                torch.cuda.empty_cache()

            dist.barrier()

    if log and cfg.local_rank == 0:
        logfile.flush()
        logfile.close()

if __name__ == '__main__':
    
    sys.path.append("configs")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="setting1_config", help="config filename")
    parser.add_argument('--log', type=str2bool, default='True', help='Whether to log')
    parser.add_argument('--exp_folder', type=str, default='FL_exp', help='sub-folder name')
    parser.add_argument('--seed', type = int, default=20220930, help = 'random seed')
    parser.add_argument('--gpu', type = int, default=0, help = 'gpu device number')
    parser.add_argument('--local_rank', type = int, default=-1, help = 'gpu device number')
    # multi_gpu 
    parser.add_argument('--use_multi_gpu', type=str2bool, default='False', help='If use multi_gpu')
    parser.add_argument('--multi_gpu', type = str, default='0', help = 'gpu device index')
    # DDP
    parser.add_argument('--nodes', default=1, type=int, metavar='node per gpu')
    parser.add_argument('--gpu_num', default=1, type=int, help='number of gpus')
    # SOLO
    available_clients = ['client_1', 'client_2', 'client_3']
    parser.add_argument('--client_name', type = str, choices=available_clients, default='client_1', help = 'client name')

    parser_args, _ = parser.parse_known_args(sys.argv)

    cfg = importlib.import_module(parser_args.config).cfg
    cfg.log = parser_args.log
    cfg.exp_folder = parser_args.exp_folder
    cfg.seed = parser_args.seed
    cfg.gpu = parser_args.gpu
    cfg.local_rank = parser_args.local_rank
    cfg.use_multi_gpu = parser_args.use_multi_gpu
    cfg.multi_gpu = parser_args.multi_gpu
    cfg.nodes = parser_args.nodes
    cfg.gpu_num = parser_args.gpu_num
    cfg.client_name = parser_args.client_name

    set_seed(cfg.seed)

    cfg.world_size = cfg.gpu_num * cfg.nodes
    cfg.dist_url = 'tcp://127.0.0.1:45411'
    mp.spawn(main, nprocs=cfg.gpu_num, args=(cfg,))
    