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
from tqdm import tqdm
from monai.transforms import (
    Compose,
    AsDiscrete,
)
import json
from utils.metric import HausdorffDistanceMetric
from monai.metrics import DiceMetric
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import copy
import warnings
warnings.filterwarnings("ignore")

from net.official_nnunet.generic_modular_UNet import *
from cmidg.imagefilter import *
from cmidg.biasfield_interpolate_cchen.adv_bias import *
from sam.minimizer import *
from my_train.all_train import *

def test(cfg, model, metric_function, post_pred, data_loader, device):

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

            data, target = data.cuda(device), target.cuda(device)

            output = sliding_window_inference(data, cfg.roi_size, cfg.sw_batch_size, model, 0.5)
   
            outputs = []
            for i in output:
                outputs.append(post_pred(i))
            outputs = torch.tensor(torch.stack(outputs), dtype=torch.uint8)

            outputs = outputs.permute([0, 2, 1, 3, 4]).flatten(0, 1)
            masks = target.permute([0, 2, 1, 3, 4]).flatten(0, 1)
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

def communication(cfg, server_model, models, client_weights, epoch, client_weights_conf=None):

    with torch.no_grad():
        for key, param in server_model.state_dict().items():
            for change_key in cfg.confidence_change_part:
                # uncertainty aware aggregation
                if cfg.confidence_change_com == True and epoch > cfg.ua_epoch and change_key in key:
                    temp = torch.zeros_like(server_model.state_dict()[key]).float()
                    for client_idx in range(len(client_weights)):
                        client_weights_conf[client_idx] = client_weights_conf[client_idx].to('cpu')
                        temp += client_weights_conf[client_idx] * models[client_idx].state_dict()[key]
                    param.copy_(temp)
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key]).float()
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    param.copy_(temp)

            for client_idx in range(len(client_weights)):
                for key_c, param_c in models[client_idx].state_dict().items():
                    if key_c == key:
                        param_c.copy_(param)
        gc.collect()

    return server_model, models

def initialize(cfg):
    train_loaders, val_loaders, test_loaders = [], [], []
    trainsets, valsets, testsets = [], [], []
    dice_metric, hausdorff_metric = [], []

    cfg.task_num = 4

    for i in range(cfg.task_num):
        dice_metric.append(DiceMetric(reduction="mean"))
        hausdorff_metric.append(HausdorffDistanceMetric(reduction="mean"))
    metric_function = [dice_metric, hausdorff_metric]

    teacher_post_pred = [
        Compose([ AsDiscrete(argmax=True, to_onehot=2) ]),
        Compose([ AsDiscrete(argmax=True, to_onehot=3) ]),
        Compose([ AsDiscrete(argmax=True, to_onehot=2) ]),
    ]
    post_pred = Compose([
        AsDiscrete(argmax=True, to_onehot=cfg.task_num + 1),
    ])

    # data sequence
    cfg.data_json_dir = cfg.data_dir + "client_data/path_json/"

    sites = ['client_1', 'client_2', 'client_3']
    cfg.client_num = len(sites)

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
    input_modalities = 1    # in_channel
    blocks_per_stage_encoder = (2, 2, 2, 2, 2, 2)
    blocks_per_stage_decoder = (2, 2, 2, 2, 2)
    feat_map_mult_on_downscale = 2
    num_classes = cfg.task_num + 1
    max_features = 320
    
    teacher_model_1 = PlainConvUNet(input_modalities, base_num_features, blocks_per_stage_encoder, feat_map_mult_on_downscale,
            pool_op_kernel_sizes, conv_op_kernel_sizes, get_default_network_config(3, dropout_p=None),
            2, blocks_per_stage_decoder, deep_supervision=True, upscale_logits=True, max_features=max_features, 
            client_num=len(sites), cfg=cfg)
    teacher_model_2 = PlainConvUNet(input_modalities, base_num_features, blocks_per_stage_encoder, feat_map_mult_on_downscale,
            pool_op_kernel_sizes, conv_op_kernel_sizes, get_default_network_config(3, dropout_p=None),
            3, blocks_per_stage_decoder, deep_supervision=True, upscale_logits=True, max_features=max_features, 
            client_num=len(sites), cfg=cfg)
    teacher_model_3 = PlainConvUNet(input_modalities, base_num_features, blocks_per_stage_encoder, feat_map_mult_on_downscale,
            pool_op_kernel_sizes, conv_op_kernel_sizes, get_default_network_config(3, dropout_p=None),
            2, blocks_per_stage_decoder, deep_supervision=True, upscale_logits=True, max_features=max_features, 
            client_num=len(sites), cfg=cfg)
    
    model = PlainConvUNet(input_modalities, base_num_features, blocks_per_stage_encoder, feat_map_mult_on_downscale,
            pool_op_kernel_sizes, conv_op_kernel_sizes, get_default_network_config(3, dropout_p=None),
            num_classes, blocks_per_stage_decoder, deep_supervision=True, upscale_logits=True, max_features=max_features, 
            client_num=len(sites), cfg=cfg)

    init_weights(model, init_type='kaiming')

    loss_fun = DiceBceMultilabelLoss(cfg)

    if cfg.sam == True:
        blender_cofig = {
            'epsilon': cfg.blend_epsilon,
            'xi': 1e-6,
            'control_point_spacing':[cfg.img_size[0]//4, cfg.img_size[1]//4, cfg.img_size[2]//4],
            'downscale':2,
            'data_size':[cfg.batch_size, 1, cfg.img_size[0], cfg.img_size[1], cfg.img_size[2]],
            'interpolation_order':1,
            'init_mode':'gaussian',
            'space':'log',
            'gpu_id': cfg.device,
        }
        blender_node = AdvBias(blender_cofig).cuda(cfg.device)
        img_transform_node = GINGroupConv(cfg, out_channel=1, n_layer=cfg.gin_nlayer, interm_channel=cfg.gin_n_interm_ch).cuda(cfg.device)

    client_num_list = []

    for site in sites:
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

    confidence_bank = []
    for _ in range(cfg.client_num):
        confidence_bank.append(Confidence_bank(cfg))

    classwise_confidence_bank = []
    for i in range(cfg.client_num):
        tmp = []
        for j in range(cfg.task_num+1):
            tmp.append(Confidence_bank(cfg))
        classwise_confidence_bank.append(tmp)

    if cfg.sam == True:
        return teacher_model_1, teacher_model_2, teacher_model_3, model, loss_fun, metric_function, teacher_post_pred, post_pred, sites, trainsets, testsets, train_loaders, val_loaders, test_loaders, client_num_list, confidence_bank, classwise_confidence_bank, blender_node, img_transform_node
    else:
        return teacher_model_1, teacher_model_2, teacher_model_3, model, loss_fun, metric_function, teacher_post_pred, post_pred, sites, trainsets, testsets, train_loaders, val_loaders, test_loaders, client_num_list, confidence_bank, classwise_confidence_bank

def main(device, cfg):
    cfg.local_rank = device

    torch.cuda.set_device(device)     
    torch.cuda.empty_cache()                 
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
    if cfg.sam == True:
        teacher_model_1, teacher_model_2, teacher_model_3, server_model, loss_fun, metric_function, teacher_post_pred, post_pred, sites, train_datasets, _, train_loaders, val_loaders, test_loaders, client_num_list, confidence_bank, classwise_confidence_bank, blender_node, img_transform_node = initialize(cfg)
    else:
        teacher_model_1, teacher_model_2, teacher_model_3, server_model, loss_fun, metric_function, teacher_post_pred, post_pred, sites, train_datasets, _, train_loaders, val_loaders, test_loaders, client_num_list, confidence_bank, classwise_confidence_bank  = initialize(cfg)
    
    if cfg.sam is True:
        global_mask = {}
        for n, p in server_model.named_parameters():
            global_mask[n] = torch.zeros_like(p)
            global_mask[n] = torch.tensor(global_mask[n], dtype=torch.uint8)
        local_masks = [global_mask for _ in range(cfg.client_num)]
        momentum_grads = []
        for _ in range(cfg.client_num):
            momentum_grad = {}
            for n, p in server_model.named_parameters():
                momentum_grad[n] = torch.zeros_like(p)
            momentum_grads.append(momentum_grad)

    print('# Deive:', cfg.multi_gpu) if cfg.local_rank == 0 else 0
    print('# Training Clients:{}'.format(sites)) if cfg.local_rank == 0 else 0

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

    # aggregation only based on num of data
    client_num = len(sites)
    num_total = sum([len(train_datasets[idx]) for idx in range(client_num)])
    client_num_weights = [float(len(train_datasets[idx])) / float(num_total) for idx in range(client_num)]

    # load pretrained model weights
    for teacher_idx in range(cfg.client_num):
        T_checkpoint = torch.load(cfg.pretrained_teacher_path[teacher_idx], map_location=torch.device('cpu'))
        if isinstance(T_checkpoint,torch.nn.DataParallel):
            T_checkpoint = T_checkpoint.module
        print('Load Client\'s {} Pretrained model:'.format(teacher_idx))
        if teacher_idx == 0:
            print(teacher_model_1.load_state_dict(T_checkpoint['Client_model']))
        elif teacher_idx == 1:
            print(teacher_model_2.load_state_dict(T_checkpoint['Client_model']))
        elif teacher_idx == 2:
            print(teacher_model_3.load_state_dict(T_checkpoint['Client_model']))

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
        server_model.load_state_dict(checkpoint['server_model'])
    
        for client_idx in range(client_num):
            models[client_idx].load_state_dict(checkpoint[f'Client{client_idx+1}_model'])

        if 'optim_0' in list(checkpoint.keys()):
            for client_idx in range(client_num):
                optimizers[client_idx].load_state_dict(checkpoint[f'optim_{client_idx}'])
                scheduler[client_idx].load_state_dict(checkpoint[f'scheduler_{client_idx}'])

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
        for client_idx, model in enumerate(models):     # training for each client
            train_loaders[client_idx].sampler.set_epoch(a_iter)
            for i in range(cfg.local_round):
                if cfg.sam == True and a_iter >= cfg.start_sam_epoch:
                    train_loss, train_dice, train_hausdorff, confidence_bank, classwise_confidence_bank, local_mask, momentum_grad \
                            = train_all(cfg, teacher_model_1, teacher_model_2, teacher_model_3, \
                            server_model, model, metric_function, teacher_post_pred, post_pred, train_loaders[client_idx], \
                            optimizers[client_idx], scheduler[client_idx], loss_fun, device, client_idx, a_iter, \
                            confidence_bank, classwise_confidence_bank, blender_node, img_transform_node, \
                            global_mask, momentum_grads[client_idx], local_masks[client_idx])
                    if a_iter == cfg.start_sam_epoch or a_iter % cfg.update_mask_fre == 1:
                        local_masks[client_idx] = local_mask
                        momentum_grads[client_idx] = momentum_grad
                else:
                    train_loss, train_dice, train_hausdorff, confidence_bank, classwise_confidence_bank = \
                        train(cfg, teacher_model_1, teacher_model_2, teacher_model_3, server_model, model, \
                        metric_function, teacher_post_pred, post_pred, train_loaders[client_idx], \
                        optimizers[client_idx], scheduler[client_idx], loss_fun, device, client_idx, a_iter, \
                        confidence_bank, classwise_confidence_bank)

                mean_dice, mean_hd = np.mean(train_dice), np.mean(train_hausdorff)
                print(' Site-{:<10s}| Train Loss: {:.4f} \n \
                        Train Dice_kidney: {:.4f}, Train Dice_liver: {:.4f}, Train Dice_spleen: {:.4f}, Train Dice_pancreas: {:.4f}, Train Dice_mean: {:.4f} \n \
                        Train HD_kidney: {:.4f}, Train HD_liver: {:.4f}, Train HD_spleen: {:.4f}, Train HD_pancreas: {:.4f}, Train HD_mean: {:.4f} '.format \
                        (sites[client_idx], train_loss, \
                            train_dice[0], train_dice[1], train_dice[2], train_dice[3], mean_dice, \
                            train_hausdorff[0], train_hausdorff[1], train_hausdorff[2], train_hausdorff[3], mean_hd))  if cfg.local_rank == 0 else 0
                if log and cfg.local_rank == 0:
                    logfile.write(' Site-{:<10s}| Train Loss: {:.4f} \n \
                        Train Dice_kidney: {:.4f}, Train Dice_liver: {:.4f}, Train Dice_spleen: {:.4f}, Train Dice_pancreas: {:.4f}, Train Dice_mean: {:.4f} \n \
                        Train HD_kidney: {:.4f}, Train HD_liver: {:.4f}, Train HD_spleen: {:.4f}, Train HD_pancreas: {:.4f}, Train HD_mean: {:.4f} '.format \
                        (sites[client_idx], train_loss, \
                            train_dice[0], train_dice[1], train_dice[2], train_dice[3], mean_dice, \
                            train_hausdorff[0], train_hausdorff[1], train_hausdorff[2], train_hausdorff[3], mean_hd))

        torch.cuda.empty_cache()

        with torch.no_grad():
            # aggregation
            dist.barrier()
            if a_iter >= cfg.com_epoch and a_iter%cfg.com_fre == 0:
                # change agg weight by uncertainty statistics
                if cfg.confidence_change_com == True and a_iter > cfg.ua_epoch:
                    client_weights_conf = update_client_weight_by_confidence(cfg, client_num_weights, confidence_bank)
                    server_model, models = communication(cfg, server_model, models, client_num_weights, a_iter, client_weights_conf)
                else:
                    server_model, models = communication(cfg, server_model, models, client_num_weights, a_iter)
                # communicate masks
                if cfg.sam is True and a_iter >= cfg.start_sam_epoch and a_iter % cfg.update_mask_fre == 1:
                    global_mask = communicate_mask(cfg, server_model, local_masks)
                gc.collect()

            # val
            if a_iter >= cfg.start_eval_epoch and np.mod(a_iter, cfg.eval_epochs) == 0:
                val_dice_list = [None for j in range(client_num)]
                val_hd_list = [None for j in range(client_num)]
                print('============== {} =============='.format('Global Validation'))  if cfg.local_rank == 0 else 0
                if log and cfg.local_rank == 0:
                    logfile.write('============== {} ==============\n'.format('Global Validation'))
                for client_idx, model in enumerate(models):
                    val_loaders[client_idx].sampler.set_epoch(a_iter)
                    val_dice, val_hausdorff = test(cfg, model, metric_function, post_pred, val_loaders[client_idx], device)
                    mean_dice, mean_hd = np.mean(val_dice), np.mean(val_hausdorff)
                    val_dice_list[client_idx] = mean_dice
                    val_hd_list[client_idx] = mean_hd
                    print(' Site-{:<10s} \n \
                            Val Dice_kidney: {:.4f}, Val Dice_liver: {:.4f}, Val Dice_spleen: {:.4f}, Val Dice_pancreas: {:.4f}, Val Dice_mean: {:.4f} \n \
                            Val HD_kidney: {:.4f}, Val HD_liver: {:.4f}, Val HD_spleen: {:.4f}, Val HD_pancreas: {:.4f}, Val HD_mean: {:.4f} '.format \
                            (sites[client_idx], \
                            val_dice[0], val_dice[1], val_dice[2], val_dice[3], mean_dice, \
                            val_hausdorff[0], val_hausdorff[1], val_hausdorff[2], val_hausdorff[3], mean_hd))  if cfg.local_rank == 0 else 0
                    if log and cfg.local_rank == 0:
                        logfile.write(' Site-{:<10s} \n \
                            Val Dice_kidney: {:.4f}, Val Dice_liver: {:.4f}, Val Dice_spleen: {:.4f}, Val Dice_pancreas: {:.4f}, Val Dice_mean: {:.4f} \n \
                            Val HD_kidney: {:.4f}, Val HD_liver: {:.4f}, Val HD_spleen: {:.4f}, Val HD_pancreas: {:.4f}, Val HD_mean: {:.4f} '.format \
                            (sites[client_idx], \
                            val_dice[0], val_dice[1], val_dice[2], val_dice[3], mean_dice, \
                            val_hausdorff[0], val_hausdorff[1], val_hausdorff[2], val_hausdorff[3], mean_hd))
                        logfile.flush()
                torch.cuda.empty_cache()

                # test
                test_dice_list = [None for j in range(client_num)]
                test_hd_list = [None for j in range(client_num)]
                print('============== {} =============='.format('Test'))  if cfg.local_rank == 0 else 0
                if log and cfg.local_rank == 0:
                    logfile.write('============== {} ==============\n'.format('Test'))
                for client_idx, model in enumerate(models):
                    test_loaders[client_idx].sampler.set_epoch(a_iter)
                    test_dice, test_hausdorff = test(cfg, model, metric_function, post_pred, test_loaders[client_idx], device)
                    mean_dice, mean_hd = np.mean(test_dice), np.mean(test_hausdorff)
                    test_dice_list[client_idx] = mean_dice
                    test_hd_list[client_idx] = mean_hd
                    print(' Test site-{:<10s}| Epoch:{} \n \
                            Test Dice_kidney: {:.4f}, Test Dice_liver: {:.4f}, Test Dice_spleen: {:.4f}, Test Dice_pancreas: {:.4f}, Test Dice_mean: {:.4f} \n \
                            Test HD_kidney: {:.4f}, Test HD_liver: {:.4f}, Test HD_spleen: {:.4f}, Test HD_pancreas: {:.4f}, Test HD_mean: {:.4f} '.format \
                            (sites[client_idx], a_iter, \
                            test_dice[0], test_dice[1], test_dice[2], test_dice[3], mean_dice, \
                            test_hausdorff[0], test_hausdorff[1], test_hausdorff[2], test_hausdorff[3], mean_hd))  if cfg.local_rank == 0 else 0
                    if log and cfg.local_rank == 0:
                        logfile.write(' Test site-{:<10s}| Epoch:{} \n \
                            Test Dice_kidney: {:.4f}, Test Dice_liver: {:.4f}, Test Dice_spleen: {:.4f}, Test Dice_pancreas: {:.4f}, Test Dice_mean: {:.4f} \n \
                            Test HD_kidney: {:.4f}, Test HD_liver: {:.4f}, Test HD_spleen: {:.4f}, Test HD_pancreas: {:.4f}, Test HD_mean: {:.4f} '.format \
                            (sites[client_idx], a_iter, \
                            test_dice[0], test_dice[1], test_dice[2], test_dice[3], mean_dice, \
                            test_hausdorff[0], test_hausdorff[1], test_hausdorff[2], test_hausdorff[3], mean_hd))
                        logfile.flush()
                torch.cuda.empty_cache()

                # record the best dice
                if np.mean(val_dice_list) > np.mean(best_dice):
                    for client_idx in range(client_num):
                        best_dice[client_idx] = val_dice_list[client_idx]
                        best_hd[client_idx] = val_hd_list[client_idx]
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
                
                    model_dicts = {'server_model': server_model.state_dict(),
                                    'Client1_model': models[0].state_dict(),
                                    'Client2_model': models[1].state_dict(),
                                    'Client3_model': models[2].state_dict(),
                                    'best_epoch': best_epoch,
                                    'best_dice': best_dice,
                                    'best_hd': best_hd,
                                    'a_iter': a_iter}
                    
                    for o_idx in range(client_num):
                        model_dicts['optim_{}'.format(o_idx)] = optimizers[o_idx].state_dict()
                        model_dicts['scheduler_{}'.format(o_idx)] = scheduler[o_idx].state_dict()
                    if dist.get_rank() == 0:
                        torch.save(model_dicts, cfg.save_path + '/best_weights.pth')
                        torch.save(model_dicts, cfg.save_path + '/latest_weights.pth')
                    best_changed = False
                else:
                    # save the latest model weights and other information
                    print(' Saving the latest checkpoint to {}...'.format(cfg.save_path))  if cfg.local_rank == 0 else 0
                    if log and cfg.local_rank == 0:
                        logfile.write(' Saving the latest checkpoint to {}...\n'.format(cfg.save_path))
                    
                    model_dicts = {'server_model': server_model.state_dict(),
                                    'Client1_model': models[0].state_dict(),
                                    'Client2_model': models[1].state_dict(),
                                    'Client3_model': models[2].state_dict(),
                                    'best_epoch': best_epoch,
                                    'best_dice': best_dice,
                                    'best_hd': best_hd,
                                    'a_iter': a_iter}
                    for o_idx in range(client_num):
                        model_dicts['optim_{}'.format(o_idx)] = optimizers[o_idx].state_dict()
                        model_dicts['scheduler_{}'.format(o_idx)] = scheduler[o_idx].state_dict()
                    if dist.get_rank() == 0:
                        torch.save(model_dicts, cfg.save_path + '/latest_weights.pth')

                del model_dicts
                gc.collect()
                torch.cuda.empty_cache()

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

    set_seed(cfg.seed)

    cfg.world_size = cfg.gpu_num * cfg.nodes
    cfg.dist_url = 'tcp://127.0.0.1:52341'
    mp.spawn(main, nprocs=cfg.gpu_num, args=(cfg,))
    