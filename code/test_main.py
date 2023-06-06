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
import pandas as pd
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from monai.transforms import (
    Compose,
    AsDiscrete,
)
import json
from monai.metrics import DiceMetric
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from scipy.ndimage import binary_fill_holes
import cc3d
import warnings
warnings.filterwarnings("ignore")

from net.official_nnunet.generic_modular_UNet import *
from utils.metric import *
from utils.utils import *
from utils.plot_util import *

# remerber to change train_site for SOLO and to comment and uncomment the metric codes in function 'test' and 'initialize'
def post_processing(vol):
    for c in range(vol.shape[0]):
        vol_ = vol[c]
        vol_ = np.array(vol_.cpu()).astype(np.int8)
        vol_ = binary_fill_holes(vol_).astype(np.int8)
        vol[c] = torch.tensor(vol_)
        
    for c in range(vol.shape[0]):
        if c != 0:
            vol_ = vol[c].cpu()
            vol_ = np.array(vol_).astype(np.int8)
            vol_cc = cc3d.connected_components(vol_)
            cc_sum = [(i, vol_cc[vol_cc == i].shape[0]) for i in range(vol_cc.max() + 1)]
            cc_sum.sort(key=lambda x: x[1], reverse=True)
            cc_sum.pop(0)  # remove background
            reduce_cc = [cc_sum[i][0] for i in range(1, len(cc_sum)) if cc_sum[i][1] < cc_sum[0][1] * 0.2]  # remove cc smaller than 20% of the largest cc (exclude background)
            for i in reduce_cc:
                vol_[vol_cc == i] = 0
            vol[c] = torch.tensor(vol_)
        else:
            continue
 
    return vol

def test(cfg, model, metric_function, post_pred, data_loader, device, client_idx):

    dice_metric, hausdorff_metric, jaccard_metric, rve_metric, sensitivity_metric, specificity_metric = metric_function 

    model = DDP(model.cuda(device), device_ids=[device], output_device=device, find_unused_parameters=True)

    torch.set_grad_enabled(False)
    progress_bar = tqdm(range(len(data_loader)))
    val_it = iter(data_loader)
    dice, hausdorff, jaccard, rve, sensitivity, specificity = [], [], [], [], [], []

    with torch.no_grad():
        for itr in progress_bar:
            model.eval()
            batch = next(val_it)

            data, target, name = (
                batch["image"],
                batch["mask"],
                batch["name"][0],
            )

            data, target = data.cuda(device), target.cuda(device)

            output = sliding_window_inference(data, cfg.roi_size, cfg.sw_batch_size, model, 0.5,
                                "constant", 0.125, "constant", 0.0, None, None, False, None)
            outputs = [post_pred(i) for i in output]
            outputs = torch.tensor(torch.stack(outputs), dtype=torch.uint8)

            # post-processing
            outputs = post_processing(outputs.squeeze(0)).unsqueeze(0).cuda(device)

            outputs = outputs.permute([0, 2, 1, 3, 4]).flatten(0, 1)
            masks = target.permute([0, 2, 1, 3, 4]).flatten(0, 1)

            if cfg.plot == True:
                # save the 3D result (this version of code only supports FL mode)
                print('saving to npy...')
                save_npy(cfg, data.permute([0, 2, 1, 3, 4]).flatten(0, 1).cpu().numpy(), masks.cpu().numpy(), outputs.cpu().numpy(), name)

            # full organ evaluation
            if cfg.test_mode == 'FL':
                for j, i in zip(range(cfg.task_num), range(1,cfg.task_num+1)):
                    hausdorff_metric[j](y_pred=outputs[:,i].unsqueeze(1), y=masks[:,i].unsqueeze(1))
                    dice_metric[j](y_pred=outputs[:,i].unsqueeze(1), y=masks[:,i].unsqueeze(1))
                    jaccard_metric[j](y_pred=outputs[:,i].unsqueeze(1), y=masks[:,i].unsqueeze(1))
                    rve_metric[j](y_pred=outputs[:,i].unsqueeze(1), y=masks[:,i].unsqueeze(1))
                    sensitivity_metric[j](y_pred=outputs[:,i].unsqueeze(1), y=masks[:,i].unsqueeze(1))
                    specificity_metric[j](y_pred=outputs[:,i].unsqueeze(1), y=masks[:,i].unsqueeze(1))
            # SOLO(partial)
            elif cfg.test_mode == 'SOLO':
                if client_idx == 0:
                    for j in range(cfg.task_num):
                        hausdorff_metric[j](y_pred=outputs[:,1].unsqueeze(1), y=masks[:,1].unsqueeze(1))
                        dice_metric[j](y_pred=outputs[:,1].unsqueeze(1), y=masks[:,1].unsqueeze(1))
                        jaccard_metric[j](y_pred=outputs[:,1].unsqueeze(1), y=masks[:,1].unsqueeze(1))
                        rve_metric[j](y_pred=outputs[:,1].unsqueeze(1), y=masks[:,1].unsqueeze(1))
                        sensitivity_metric[j](y_pred=outputs[:,1].unsqueeze(1), y=masks[:,1].unsqueeze(1))
                        specificity_metric[j](y_pred=outputs[:,1].unsqueeze(1), y=masks[:,1].unsqueeze(1))
                elif client_idx == 1:
                    for j in range(cfg.task_num):
                        hausdorff_metric[j](y_pred=outputs[:,j+1].unsqueeze(1), y=masks[:,j+3].unsqueeze(1))
                        dice_metric[j](y_pred=outputs[:,j+1].unsqueeze(1), y=masks[:,j+3].unsqueeze(1))
                        jaccard_metric[j](y_pred=outputs[:,j+1].unsqueeze(1), y=masks[:,j+3].unsqueeze(1))
                        rve_metric[j](y_pred=outputs[:,j+1].unsqueeze(1), y=masks[:,j+3].unsqueeze(1))
                        sensitivity_metric[j](y_pred=outputs[:,j+1].unsqueeze(1), y=masks[:,j+3].unsqueeze(1))
                        specificity_metric[j](y_pred=outputs[:,j+1].unsqueeze(1), y=masks[:,j+3].unsqueeze(1))
                elif client_idx == 2:
                    for j in range(cfg.task_num):
                        hausdorff_metric[j](y_pred=outputs[:,1].unsqueeze(1), y=masks[:,2].unsqueeze(1))
                        dice_metric[j](y_pred=outputs[:,1].unsqueeze(1), y=masks[:,2].unsqueeze(1))
                        jaccard_metric[j](y_pred=outputs[:,1].unsqueeze(1), y=masks[:,2].unsqueeze(1))
                        rve_metric[j](y_pred=outputs[:,1].unsqueeze(1), y=masks[:,2].unsqueeze(1))
                        sensitivity_metric[j](y_pred=outputs[:,1].unsqueeze(1), y=masks[:,2].unsqueeze(1))
                        specificity_metric[j](y_pred=outputs[:,1].unsqueeze(1), y=masks[:,2].unsqueeze(1))

    for i in range(cfg.task_num):
        dice.append(dice_metric[i].aggregate().item())
        hausdorff.append(hausdorff_metric[i].aggregate().item())
        jaccard.append(jaccard_metric[i].aggregate().item())
        rve.append(rve_metric[i].aggregate().item())
        sensitivity.append(sensitivity_metric[i].aggregate().item())
        specificity.append(specificity_metric[i].aggregate().item())

        dice_metric[i].reset()
        hausdorff_metric[i].reset()
        jaccard_metric[i].reset()
        rve_metric[i].reset()
        sensitivity_metric[i].reset()
        specificity_metric[i].reset()

    model.to('cpu')

    del data, target, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return dice, hausdorff, jaccard, rve, sensitivity, specificity

def initialize(cfg):
    testsets, test_loaders = [], []
    dice_metric, hausdorff_metric, jaccard_metric, rve_metric, sensitivity_metric, specificity_metric = [], [], [], [], [], []

    # choose only one for SOLO, others use all
    if cfg.train_site == 'all':
        train_site = ['client_1', 'client_2', 'client_3']
    else:
        train_site = cfg.train_site

    client_idx = 0
    if cfg.test_mode == 'SOLO':
        if 'client_1' == train_site:
            client_idx = 0
            cfg.task_num = 1    
        elif 'client_2' == train_site:
            client_idx = 1
            cfg.task_num = 2
        elif 'client_3' == train_site:
            client_idx = 2
            cfg.task_num = 1

    for i in range(cfg.task_num):
        dice_metric.append(DiceMetric(reduction="mean"))
        hausdorff_metric.append(HausdorffDistanceMetric(reduction="mean"))
        jaccard_metric.append(JaccardCoefficientMetric(reduction="mean"))
        rve_metric.append(RelativeVolumeErrorMetric(reduction="mean"))
        sensitivity_metric.append(SensitivityMetric(reduction="mean"))
        specificity_metric.append(SpecificityMetric(reduction="mean"))

    metric_function = [dice_metric, hausdorff_metric, jaccard_metric, rve_metric, sensitivity_metric, specificity_metric]

    post_pred = Compose([
        AsDiscrete(argmax=True, to_onehot=cfg.task_num+1),
    ])

    # data sequence
    cfg.data_json_dir = cfg.data_dir + "client_data/path_json/"

    sites = ['client_1', 'client_2', 'client_3', 'client_4']
    
    if cfg.test_mode == 'FL':
        cfg.client_num = 3
    elif cfg.test_mode == 'SOLO':
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
    base_num_features = 32
    input_modalities = 1
    blocks_per_stage_encoder = (2, 2, 2, 2, 2, 2)
    blocks_per_stage_decoder = (2, 2, 2, 2, 2)
    feat_map_mult_on_downscale = 2
    num_classes = cfg.task_num+1 
    max_features = 320

    model = PlainConvUNet(input_modalities, base_num_features, blocks_per_stage_encoder, feat_map_mult_on_downscale,
        pool_op_kernel_sizes, conv_op_kernel_sizes, get_default_network_config(3, dropout_p=None),
        num_classes, blocks_per_stage_decoder, deep_supervision=True, upscale_logits=True, max_features=max_features, 
        client_num=cfg.client_num, cfg=cfg)
            
    init_weights(model, init_type='kaiming')

    for site in sites:
        with open(cfg.data_json_dir + site + '.json', "r") as f:
            cfg.data_json = json.load(f)

        testset = get_test_dataset(cfg)
        print(f'[Client {site[-1]}] Test={len(testset)}') if cfg.local_rank == 0 else 0

        testsets.append(testset)

    for idx in range(len(testsets)):
        testset = testsets[idx]
        test_loaders.append(get_val_dataloader(testset, cfg))
        
    return model, metric_function, post_pred, sites, test_loaders, client_idx

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
    if not os.path.exists(cfg.plot_path) and cfg.local_rank == 0:
        os.makedirs(cfg.plot_path)

    cfg.device = device

    server_model, metric_function, post_pred, sites, test_loaders, client_idx  = initialize(cfg)

    print('# Deive:', cfg.multi_gpu) if cfg.local_rank == 0 else 0
    print('# Testing Clients:{}'.format(sites)) if cfg.local_rank == 0 else 0

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
    
    pd_data = []

    models = [copy.deepcopy(server_model) for _ in range(cfg.client_num)]
    print('Loading snapshots...') if cfg.local_rank == 0 else 0
    # SOLO(partial)
    if cfg.test_mode == 'SOLO':
        for i in range(cfg.client_num):
            print(models[i].load_state_dict(torch.load(cfg.test_path)['Client_model']))
    # FL
    elif cfg.test_mode == 'FL':
        checkpoint = torch.load(cfg.test_path)
        print(server_model.load_state_dict(checkpoint['server_model']))

    # SOLO(partial)
    if cfg.test_mode == 'SOLO':
        for _, model in enumerate(models):
            pd_data = []
            for dataset_idx, test_loader in enumerate(test_loaders):
                test_dice, test_hausdorff, test_jaccard, test_rve, test_sensitivity, test_specificity = test(cfg, model, metric_function, post_pred, test_loader, device, client_idx)
                a_iter = -1
                if client_idx == 0:
                    print(' Test site-{:<10s}| Epoch:{} \n \
                            Test Dice_kidney: {:.4f} \n \
                            Test HD_kidney: {:.4f}  \n \
                            Test jaccard_kidney: {:.4f}  \n \
                            Test rve_kidney: {:.4f}  \n \
                            Test sensitivity_kidney: {:.4f}  \n \
                            Test specificity_kidney: {:.4f}  \n \ '.format \
                            (sites[dataset_idx], a_iter, \
                            test_dice[0]*100,\
                            test_hausdorff[0],\
                            test_jaccard[0],\
                            test_rve[0],\
                            test_sensitivity[0],\
                            test_specificity[0]))  if cfg.local_rank == 0 else 0
                    client_pd_data = \
                        [
                            test_dice[0]*100, \
                            test_hausdorff[0],\
                            test_jaccard[0],\
                            test_rve[0],\
                            test_sensitivity[0],\
                            test_specificity[0]
                        ]
                    pd_data.append(client_pd_data)
                elif client_idx == 1:
                    print(' Test site-{:<10s}| Epoch:{} \n \
                            Test Dice_Spleen: {:.4f}, Test Dice_Pancreas: {:.4f}\n \
                            Test HD_Spleen: {:.4f}, Test HD_Pancreas: {:.4f}\n \
                            Test jaccard_Spleen: {:.4f}, Test jaccard_Pancreas: {:.4f}\n \
                            Test rve_Spleen: {:.4f}, Test rve_Pancreas: {:.4f}\n \
                            Test sensitivity_Spleen: {:.4f}, Test sensitivity_Pancreas: {:.4f}\n \
                            Test specificity_Spleen: {:.4f}, Test specificity_Pancreas: {:.4f}\n \ '.format \
                            (sites[dataset_idx], a_iter, \
                            test_dice[0]*100, test_dice[1]*100,\
                            test_hausdorff[0], test_hausdorff[1],\
                            test_jaccard[0], test_jaccard[1],\
                            test_rve[0], test_rve[1],\
                            test_sensitivity[0], test_sensitivity[1],\
                            test_specificity[0], test_specificity[1]  ))  if cfg.local_rank == 0 else 0
                    client_pd_data = \
                        [
                            test_dice[0]*100, test_dice[1]*100, \
                            test_hausdorff[0], test_hausdorff[1],\
                            test_jaccard[0], test_jaccard[1],\
                            test_rve[0], test_rve[1],\
                            test_sensitivity[0], test_sensitivity[1],\
                            test_specificity[0], test_specificity[1]
                        ]
                    pd_data.append(client_pd_data)
                elif client_idx == 2:
                    print(' Test site-{:<10s}| Epoch:{} \n \
                            Test Dice_Lung: {:.4f} \n \
                            Test HD_Lung: {:.4f}  \n \
                            Test jaccard_Lung: {:.4f}  \n \
                            Test rve_Lung: {:.4f}  \n \
                            Test sensitivity_Lung: {:.4f}  \n \
                            Test specificity_Lung: {:.4f}  \n \ '.format \
                            (sites[dataset_idx], a_iter, \
                            test_dice[0]*100,\
                            test_hausdorff[0],\
                            test_jaccard[0],\
                            test_rve[0],\
                            test_sensitivity[0],\
                            test_specificity[0]))  if cfg.local_rank == 0 else 0
                    client_pd_data = \
                        [
                            test_dice[0]*100, \
                            test_hausdorff[0],\
                            test_jaccard[0],\
                            test_rve[0],\
                            test_sensitivity[0],\
                            test_specificity[0]
                        ]
                    pd_data.append(client_pd_data)
                    
        if client_idx == 0:
            names=["client1", "client2", "client3", 'client4']
            exems=["dice", "hd", "jc", "rve", "sensitivity", "specificity"]
            index=pd.MultiIndex.from_product([names,exems], names=["client_idx", "metric"])
            columns = ['kidney']
            pd_data = np.array(pd_data).reshape(4*6, 1)
            df = pd.DataFrame(pd_data, index=index, columns=columns).round(3)
            df_name = cfg.test_path.split('/')[-2]
            df.to_csv(f'./{df_name}_solo_client{client_idx}.csv')
        elif client_idx == 1:
            names=["client1", "client2", "client3", 'client4']
            exems=["dice", "hd", "jc", "rve", "sensitivity", "specificity"]
            index=pd.MultiIndex.from_product([names,exems], names=["client_idx", "metric"])
            columns = ['spleen', 'pancreas']
            pd_data = np.array(pd_data).reshape(4*6, 2)
            df = pd.DataFrame(pd_data, index=index, columns=columns).round(3)
            df_name = cfg.test_path.split('/')[-2]
            df.to_csv(f'./{df_name}_solo_client{client_idx}.csv')
        elif client_idx == 2:
            names=["client1", "client2", "client3", 'client4']
            exems=["dice", "hd", "jc", "rve", "sensitivity", "specificity"]
            index=pd.MultiIndex.from_product([names,exems], names=["client_idx", "metric"])
            columns = ['liver']
            pd_data = np.array(pd_data).reshape(4*6, 1)
            df = pd.DataFrame(pd_data, index=index, columns=columns).round(3)
            df_name = cfg.test_path.split('/')[-2]
            df.to_csv(f'./{df_name}_solo_client{client_idx}.csv')
    # FL
    elif cfg.test_mode == 'FL':
        for dataset_idx, site in enumerate(sites):
            test_dice, test_hausdorff, test_jaccard, test_rve, test_sensitivity, test_specificity = test(cfg, server_model, metric_function, post_pred, test_loaders[dataset_idx], device, 0)
            mean_dice, mean_hd, mean_jaccard, mean_rve, mean_sensitivity, mean_specificity = np.mean(test_dice), np.mean(test_hausdorff), np.mean(test_jaccard), np.mean(test_rve), np.mean(test_sensitivity), np.mean(test_specificity)

            print(' Test_site: {} \n \
                    Test Dice_kidney: {:.4f}, Test Dice_liver: {:.4f}, Test Dice_spleen: {:.4f}, Test Dice_pancreas: {:.4f}, Test Dice_mean: {:.4f} \n \
                    Test HD_kidney: {:.4f}, Test HD_liver: {:.4f}, Test HD_spleen: {:.4f}, Test HD_pancreas: {:.4f}, Test HD_mean: {:.4f}  \n \
                    Test jaccard_kidney: {:.4f}, Test jaccard_liver: {:.4f}, Test jaccard_spleen: {:.4f}, Test jaccard_pancreas: {:.4f}, Test jaccard_mean: {:.4f}  \n \
                    Test rve_kidney: {:.4f}, Test rve_liver: {:.4f}, Test rve_spleen: {:.4f}, Test rve_pancreas: {:.4f}, Test rve_mean: {:.4f}  \n \
                    Test sensitivity_kidney: {:.4f}, Test sensitivity_liver: {:.4f}, Test sensitivity_spleen: {:.4f}, Test sensitivity_pancreas: {:.4f}, Test sensitivity_mean: {:.4f}  \n \
                    Test specificity_kidney: {:.4f}, Test specificity_liver: {:.4f}, Test specificity_spleen: {:.4f}, Test specificity_pancreas: {:.4f}, Test specificity_mean: {:.4f}  \n \ '.format \
                    (
                    sites[dataset_idx],
                    test_dice[0]*100, test_dice[1]*100, test_dice[2]*100, test_dice[3]*100, mean_dice*100, \
                    test_hausdorff[0], test_hausdorff[1], test_hausdorff[2], test_hausdorff[3], mean_hd,\
                    test_jaccard[0], test_jaccard[1], test_jaccard[2], test_jaccard[3], mean_jaccard,\
                    test_rve[0], test_rve[1], test_rve[2], test_rve[3], mean_rve,\
                    test_sensitivity[0], test_sensitivity[1], test_sensitivity[2], test_sensitivity[3], mean_sensitivity,\
                    test_specificity[0], test_specificity[1], test_specificity[2], test_specificity[3], mean_specificity))  if cfg.local_rank == 0 else 0

            client_pd_data = \
                [
                    test_dice[0]*100, test_dice[1]*100, test_dice[2]*100, test_dice[3]*100, mean_dice*100, \
                    test_hausdorff[0], test_hausdorff[1], test_hausdorff[2], test_hausdorff[3], mean_hd,\
                    test_jaccard[0], test_jaccard[1], test_jaccard[2], test_jaccard[3], mean_jaccard,\
                    test_rve[0], test_rve[1], test_rve[2], test_rve[3], mean_rve,\
                    test_sensitivity[0], test_sensitivity[1], test_sensitivity[2], test_sensitivity[3], mean_sensitivity,\
                    test_specificity[0], test_specificity[1], test_specificity[2], test_specificity[3], mean_specificity
                ]
            pd_data.append(client_pd_data)

        names=["client1", "client2", "client3", 'client4']
        exems=["dice", "hd", "jc", "rve", "sensitivity", "specificity"]
        index=pd.MultiIndex.from_product([names,exems], names=["client_idx", "metric"])
        columns = ['kidney', 'liver', 'spleen', 'pancreas','mean']
        pd_data = np.array(pd_data).reshape(4*6, 5)
        df = pd.DataFrame(pd_data, index=index, columns=columns).round(3)
        df_name = cfg.test_path.split('/')[-2]
        df.to_csv(f'./{df_name}.csv')

        if log and cfg.local_rank == 0:
            logfile.flush()
            logfile.close()


if __name__ == '__main__':
    
    sys.path.append("configs")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="setting1_config", help="config filename")
    parser.add_argument('--log', type=str2bool, default='True', help='Whether to log')
    parser.add_argument('--exp_folder', type=str, default='FL_exp', help='sub-folder name')
    parser.add_argument('--seed', type = int, default=0, help = 'random seed')
    parser.add_argument('--gpu', type = int, default=0, help = 'gpu device number')
    parser.add_argument('--local_rank', type = int, default=-1, help = 'gpu device number')
    # multi_gpu 
    parser.add_argument('--use_multi_gpu', type=str2bool, default='False', help='If use multi_gpu')
    parser.add_argument('--multi_gpu', type = str, default='0,1', help = 'gpu device index')
    # DDP
    parser.add_argument('--nodes', default=1, type=int, metavar='node per gpu')
    parser.add_argument('--gpu_num', default=2, type=int, help='number of gpus')
    # vis
    parser.add_argument('--plot', type=str2bool, default='False', help='If plot result')
    parser.add_argument('--plot_path', type = str, default='../plot_result/', help='path to save plotting result')
    # test mode
    available_test_mode = ['SOLO', 'FL']
    parser.add_argument('--test_mode', type = str, choices=available_test_mode, default='SOLO', help = 'test mode')
    available_clients = ['client_1', 'client_2', 'client_3', 'all']
    parser.add_argument('--train_site', type = str, choices=available_clients, default='client_1', help = 'pretrained sites')

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
    cfg.plot = parser_args.plot
    cfg.plot_path = parser_args.plot_path
    cfg.test_mode = parser_args.test_mode
    cfg.train_site = parser_args.train_site

    set_seed(cfg.seed)

    cfg.world_size = cfg.gpu_num * cfg.nodes
    cfg.dist_url = 'tcp://127.0.0.1:29876'
    mp.spawn(main, nprocs=cfg.gpu_num, args=(cfg,))
    
    