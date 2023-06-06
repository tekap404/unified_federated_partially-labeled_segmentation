from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import sys
sys.path.append('../')
from utils.utils import *
from sam.minimizer import *
from cmidg.imagefilter import *
from cmidg.biasfield_interpolate_cchen.adv_bias import *

def get_teacher_pred(cfg, data, target, teacher_model_1, teacher_model_2, teacher_model_3, \
                    teacher_post_pred, client_idx, confidence_bank=None , classwise_confidence_bank=None):

    with torch.no_grad():
        # get pred from teachers
        logits_teacher_all = torch.zeros_like(target).to(target.device)
        outputs_teacher_all = torch.tensor(torch.zeros_like(target), dtype=torch.int8).to(target.device)
        for i in range(3):
            if i == 0:
                logits_teacher = teacher_model_1(data)
            elif i == 1:
                logits_teacher = teacher_model_2(data)
            elif i == 2:
                logits_teacher = teacher_model_3(data)
            output_teacher = []
            for a in logits_teacher:
                output_teacher.append(teacher_post_pred[i](a))
            outputs_teacher = torch.tensor(torch.stack(output_teacher), dtype=torch.int8)

            # merge teacher pred
            logits_teacher_all, outputs_teacher_all = mix_all_teacher_preds(logits_teacher, outputs_teacher, i, logits_teacher_all, outputs_teacher_all)

        del logits_teacher, outputs_teacher
        gc.collect()
        torch.cuda.empty_cache()

        # intersection of background pred from all teachers as final background
        outputs_teacher_all[:,0] = torch.where(outputs_teacher_all[:,0]<3, 0, 1)
        logits_teacher_all[:,0] = logits_teacher_all[:,0] / 3

        # get entropy and deposit into uncertainty bank
        if cfg.cal_confidence_weight == True:
            confidence_score = []
            for bs in range(len(logits_teacher_all)):
                confidence = get_confidence(logits_teacher_all[bs], outputs_teacher_all[bs])
                confidence_score.append(confidence)
                confidence_bank[client_idx].push(confidence.cpu())
                for c in range(cfg.task_num+1):
                    classwise_confidence = get_classwise_confidence(logits_teacher_all[bs,c], outputs_teacher_all[bs,c])
                    classwise_confidence_bank[client_idx][c].push(classwise_confidence.cpu())
            confidence_score = torch.tensor(torch.stack(confidence_score), dtype=torch.float32)
            del logits_teacher_all
            gc.collect()
            torch.cuda.empty_cache()

        # replace the channel(s) with GT at hand
        if client_idx == 0:
            outputs_teacher_all[:,1] = target[:,1] 
        elif client_idx == 1:
            outputs_teacher_all[:,3:] = target[:,3:]
        elif client_idx == 2:
            outputs_teacher_all[:,2] = target[:,2]

    if cfg.cal_confidence_weight == True:
        return confidence_score, outputs_teacher_all
    else:
        return outputs_teacher_all

def get_global_model_pred(cfg, data, target, server_model, teacher_model_1, teacher_model_2, teacher_model_3, \
                    post_pred, teacher_post_pred, client_idx, confidence_bank=None , classwise_confidence_bank=None):

    with torch.no_grad():
        # get pred from the global model
        outputs_teacher_all = torch.tensor(torch.zeros_like(target), dtype=torch.int8).to(target.device)
        logits_teacher_all = server_model(data)
        outputs_teacher_all = []
        for i in logits_teacher_all:
            outputs_teacher_all.append(post_pred(i))
        outputs_teacher_all = torch.tensor(torch.stack(outputs_teacher_all), dtype=torch.uint8)
        
        # get entropy and deposit into uncertainty bank
        if cfg.cal_confidence_weight == True:
            confidence_score = []
            for bs in range(len(logits_teacher_all)):
                confidence = get_confidence(logits_teacher_all[bs], outputs_teacher_all[bs])
                confidence_score.append(confidence)
                confidence_bank[client_idx].push(confidence.cpu())
                for c in range(cfg.task_num+1):
                    classwise_confidence = get_classwise_confidence(logits_teacher_all[bs,c], outputs_teacher_all[bs,c])
                    classwise_confidence_bank[client_idx][c].push(classwise_confidence.cpu())
            confidence_score = torch.tensor(torch.stack(confidence_score), dtype=torch.float32)

        # get pred from pretrained teachers
        logits_oralteacher_all = torch.zeros_like(target).to(target.device)
        outputs_oralteacher_all = torch.tensor(torch.zeros_like(target), dtype=torch.int8).to(target.device)
        for i in range(3):
            if i == 0:
                logits_teacher = teacher_model_1(data)
            elif i == 1:
                logits_teacher = teacher_model_2(data)
            elif i == 2:
                logits_teacher = teacher_model_3(data)
            output_teacher = []
            for a in logits_teacher:
                output_teacher.append(teacher_post_pred[i](a))
            outputs_teacher = torch.tensor(torch.stack(output_teacher), dtype=torch.int8)

            # merge teacher pred
            logits_oralteacher_all, outputs_oralteacher_all = mix_all_teacher_preds(logits_teacher, outputs_teacher, i, logits_oralteacher_all, outputs_oralteacher_all)

        del logits_teacher, outputs_teacher
        gc.collect()
        torch.cuda.empty_cache()

        # intersection of background pred from all teachers as final background
        outputs_oralteacher_all[:,0] = torch.where(outputs_oralteacher_all[:,0]<3, 0, 1)
        logits_oralteacher_all[:,0] = logits_oralteacher_all[:,0] / 3

        # get the intersectin between preds from the global one and pretrained one or the original global pred
        for bs in range(len(outputs_teacher_all)):
            outputs_teacher_all[bs] = get_teachers_intersection(cfg, outputs_teacher_all[bs], outputs_oralteacher_all[bs])
        del outputs_oralteacher_all
        gc.collect()
        torch.cuda.empty_cache()

        # replace the channel(s) with GT at hand
        if client_idx == 0:
            outputs_teacher_all[:,1] = target[:,1] 
        elif client_idx == 1:
            outputs_teacher_all[:,3:] = target[:,3:]
        elif client_idx == 2:
            outputs_teacher_all[:,2] = target[:,2]

    if cfg.cal_confidence_weight == True:
        return confidence_score, outputs_teacher_all
    else:
        return outputs_teacher_all

def compute_loss(cfg, a_iter, loss_fun, output, outputs_teacher_all, confidence_score=None, confidence_statics=None):

    if cfg.rce == True and a_iter > cfg.final_weight_epoch:
        rce_flag = True
    else:
        rce_flag = False

    # change loss according to entropy
    if cfg.confidence_weight_loss == True and a_iter > cfg.warmup_epoch and a_iter <= cfg.final_weight_epoch:
        confidence_weight = get_confidence_weight(cfg, confidence_score, confidence_statics, a_iter)
        loss = loss_fun(output, torch.tensor(outputs_teacher_all, dtype=output.dtype), rce_flag, a_iter, confidence_weight)
    else:
        loss = loss_fun(output, torch.tensor(outputs_teacher_all, dtype=output.dtype), rce_flag, a_iter)

    return loss

def compute_metric(cfg, output, target, post_pred, hausdorff_metric, dice_metric):
    outputs = []
    for i in output[:,4]:
        outputs.append(post_pred(i))
    outputs = torch.tensor(torch.stack(outputs), dtype=torch.uint8)
    # metric (n*d, c, h, w)
    outputs = outputs.permute([0, 2, 1, 3, 4]).flatten(0, 1)
    masks = target.permute([0, 2, 1, 3, 4]).flatten(0, 1)
    for j, i in zip(range(cfg.task_num), range(1,cfg.task_num+1)):
        hausdorff_metric[j](y_pred=outputs[:,i].unsqueeze(1), y=masks[:,i].unsqueeze(1))
        dice_metric[j](y_pred=outputs[:,i].unsqueeze(1), y=masks[:,i].unsqueeze(1))

def get_cmidg_data(cfg, data, img_transform_node, blender_node, device):

    _nb_current = data.shape[0]
    # gin (bs*2)
    input_buffer = torch.cat([img_transform_node(data, device) for _ in range(2)], dim = 0)
    # ipa
    blender_node.init_parameters()
    blend_mask = rescale_intensity(blender_node.bias_field).repeat(1,1,1,1,1)
    # merge
    input_cp1 = input_buffer[: _nb_current].clone().detach() * blend_mask + input_buffer[_nb_current: _nb_current * 2].clone().detach() * (1.0 - blend_mask)
    input_cp1 = input_cp1.to(cfg.device)

    return input_cp1

def communicate_mask(cfg, server_model, local_masks):

    with torch.no_grad():
        if 'module' in list(local_masks[0].keys())[0]:
            prex = 'module.'
        else:
            prex = ''

        # initialization
        global_mask = {}
        for n, p in server_model.named_parameters():
            global_mask[n] = torch.zeros_like(p)
            global_mask[n] = torch.tensor(global_mask[n], dtype=torch.uint8)
        
        for n, p in server_model.named_parameters():
            for client_idx in range(cfg.client_num):
                global_mask[n] += local_masks[client_idx][prex+n]
            # not the intersection for all clients = 0
            global_mask[n] = torch.where((global_mask[n] == cfg.client_num) | (global_mask[n] == 0), 1, 0)
            global_mask[n].require_grad = False
            
        gc.collect()

    return global_mask

# without sam
def train(cfg, teacher_model_1, teacher_model_2, teacher_model_3, server_model, model, \
            metric_function, teacher_post_pred, post_pred, data_loader, \
            optimizer, scheduler, loss_fun, device, client_idx, a_iter, \
            confidence_bank, classwise_confidence_bank):

    dice_metric, hausdorff_metric = metric_function
    
    model = DDP(model.cuda(device), device_ids=[device], output_device=device, find_unused_parameters=True)
    model.train()
    if cfg.global_pseduo == True and a_iter >= cfg.global_pseduo_epoch:
        server_model = DDP(server_model.cuda(device), device_ids=[device], output_device=device)
        server_model.eval()
    teacher_model_1 = DDP(teacher_model_1.cuda(device), device_ids=[device], output_device=device)
    teacher_model_1.eval()
    teacher_model_2 = DDP(teacher_model_2.cuda(device), device_ids=[device], output_device=device)
    teacher_model_2.eval()
    teacher_model_3 = DDP(teacher_model_3.cuda(device), device_ids=[device], output_device=device)
    teacher_model_3.eval()
    
    scaler = GradScaler(init_scale=2.**8)

    progress_bar = tqdm(range(len(data_loader)))
    tr_it = iter(data_loader)

    loss_all = 0
    dataset_size = 0
    running_loss = 0.0
    len_data_loader = len(data_loader)
    dice, hausdorff = [], []
    if cfg.cal_confidence_weight == True and a_iter > cfg.warmup_epoch:
        confidence_statics = confidence_bank[client_idx].get_statics()
        classwise_confidence_statics = []
        for c in range(cfg.task_num+1):
            classwise_confidence_statics.append(classwise_confidence_bank[client_idx][c].mean)
    dist.barrier()

    for itr in progress_bar:
        batch = next(tr_it)
        optimizer.zero_grad()

        data, target, data_name = (
            batch["image"],
            batch["mask"],
            batch["name"][0],
        )

        data, target = data.cuda(device), target.cuda(device)
            
        with autocast(enabled=True):

            # get pred from the global model / intersection between global and pretrained models
            if cfg.global_pseduo == True and a_iter >= cfg.global_pseduo_epoch:
                if cfg.cal_confidence_weight == True:
                    confidence_score, outputs_teacher_all = get_global_model_pred(cfg, data, target, server_model, \
                                teacher_model_1, teacher_model_2, teacher_model_3, \
                                post_pred, teacher_post_pred, client_idx, confidence_bank, classwise_confidence_bank)
                else:
                    outputs_teacher_all = get_global_model_pred(cfg, data, target, server_model, \
                                            teacher_model_1, teacher_model_2, teacher_model_3, \
                                            post_pred, teacher_post_pred, client_idx)
            # get pred from pretrained teachers
            else:
                if cfg.cal_confidence_weight == True:
                    confidence_score, outputs_teacher_all = get_teacher_pred(cfg, data, target, \
                                teacher_model_1, teacher_model_2, teacher_model_3, \
                                teacher_post_pred, client_idx, confidence_bank, classwise_confidence_bank)
                else:
                    outputs_teacher_all = get_teacher_pred(cfg, data, target, \
                                teacher_model_1, teacher_model_2, teacher_model_3, \
                                teacher_post_pred, client_idx)
            
            output = model(data)

            # loss
            if cfg.cal_confidence_weight == True and a_iter > cfg.warmup_epoch:
                loss = compute_loss(cfg, a_iter, loss_fun, output, outputs_teacher_all, confidence_score=confidence_score, confidence_statics=confidence_statics)
            else:
                loss = compute_loss(cfg, a_iter, loss_fun, output, outputs_teacher_all)

            dist.barrier()  # synchronizes all processes
            dist.all_reduce(loss, op=torch.distributed.ReduceOp.AVG,)  # get mean loss for all processes

            loss_all = loss_all + loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(optimizer)
            scaler.update()
            
            # calculate mertic
            compute_metric(cfg, output, target, post_pred, hausdorff_metric, dice_metric)

        running_loss = running_loss + (loss.item() * cfg.batch_size)
        dataset_size = dataset_size + cfg.batch_size
        
        scheduler.step()
    
    dist.barrier()
    for i in range(cfg.task_num):
        dice.append(dice_metric[i].aggregate().item())
        hausdorff.append(hausdorff_metric[i].aggregate().item())
        dice_metric[i].reset()
        hausdorff_metric[i].reset()

    if cfg.cal_confidence_weight == True:
        confidence_bank[client_idx].aggregate()
        for c in range(cfg.task_num+1):
            classwise_confidence_bank[client_idx][c].update_mean()
            classwise_confidence_bank[client_idx][c].reset()

    loss = loss_all / len_data_loader

    model.to('cpu')
    server_model.to('cpu')
    teacher_model_1.to('cpu')
    teacher_model_2.to('cpu')
    teacher_model_3.to('cpu')
    if cfg.global_pseduo == True and a_iter >= cfg.global_pseduo_epoch:
        server_model.to('cpu')
    
    del data, target, loss_all, loss_fun, output, running_loss, outputs_teacher_all
    gc.collect()
    torch.cuda.empty_cache()
    
    return loss, dice, hausdorff, confidence_bank, classwise_confidence_bank

# with sam
def train_all(cfg, teacher_model_1, teacher_model_2, teacher_model_3, server_model, model, metric_function, \
                teacher_post_pred, post_pred, data_loader, optimizer, scheduler, loss_fun, device, client_idx, \
                a_iter, confidence_bank, classwise_confidence_bank, blender_node, img_transform_node, \
                global_mask=None, momentum_grad=None, local_mask=None):

    dice_metric, hausdorff_metric = metric_function
    
    model = DDP(model.cuda(device), device_ids=[device], output_device=device, find_unused_parameters=True)
    model.train()
    if cfg.global_pseduo == True and a_iter >= cfg.global_pseduo_epoch:
        server_model = DDP(server_model.cuda(device), device_ids=[device], output_device=device)
        server_model.eval()
    teacher_model_1 = DDP(teacher_model_1.cuda(device), device_ids=[device], output_device=device)
    teacher_model_1.eval()
    teacher_model_2 = DDP(teacher_model_2.cuda(device), device_ids=[device], output_device=device)
    teacher_model_2.eval()
    teacher_model_3 = DDP(teacher_model_3.cuda(device), device_ids=[device], output_device=device)
    teacher_model_3.eval()
    
    scaler = GradScaler(init_scale=2.**8)

    progress_bar = tqdm(range(len(data_loader)))
    tr_it = iter(data_loader)

    loss_all = 0
    dataset_size = 0
    running_loss = 0.0
    len_data_loader = len(data_loader)
    dice, hausdorff = [], []
    if cfg.cal_confidence_weight == True and a_iter > cfg.warmup_epoch:
        confidence_statics = confidence_bank[client_idx].get_statics()
        classwise_confidence_statics = []
        for c in range(cfg.task_num+1):
            classwise_confidence_statics.append(classwise_confidence_bank[client_idx][c].mean)
    dist.barrier()

    minimizer = ASAM(optimizer, model, cfg.rho, cfg.eta, cfg, momentum_grad)

    for itr in progress_bar:
        batch = next(tr_it)
        optimizer.zero_grad()

        data, target, data_name = (
            batch["image"],
            batch["mask"],
            batch["name"][0],
        )

        data, target = data.cuda(device), target.cuda(device)
            
        with autocast(enabled=True):

            if cfg.global_pseduo == True and a_iter >= cfg.global_pseduo_epoch:
                if cfg.cal_confidence_weight == True:
                    confidence_score, outputs_teacher_all = get_global_model_pred(cfg, data, target, server_model, \
                                teacher_model_1, teacher_model_2, teacher_model_3, \
                                post_pred, teacher_post_pred, client_idx, confidence_bank, classwise_confidence_bank)
                else:
                    outputs_teacher_all = get_global_model_pred(cfg, data, target, server_model, \
                                            teacher_model_1, teacher_model_2, teacher_model_3, \
                                            post_pred, teacher_post_pred, client_idx)
            else:
                if cfg.cal_confidence_weight == True:
                    confidence_score, outputs_teacher_all = get_teacher_pred(cfg, data, target, \
                                teacher_model_1, teacher_model_2, teacher_model_3, \
                                teacher_post_pred, client_idx, confidence_bank, classwise_confidence_bank)
                else:
                    outputs_teacher_all = get_teacher_pred(cfg, data, target, \
                                teacher_model_1, teacher_model_2, teacher_model_3, \
                                teacher_post_pred, client_idx)
            
            # generate CMIDG augmented data
            input_cp1 = get_cmidg_data(cfg, data, img_transform_node, blender_node, device)

            output = model(input_cp1)

            # loss
            if cfg.cal_confidence_weight == True:
                loss = compute_loss(cfg, a_iter, loss_fun, output, outputs_teacher_all, confidence_score=confidence_score, confidence_statics=confidence_statics)
            else:
                loss = compute_loss(cfg, a_iter, loss_fun, output, outputs_teacher_all)

            dist.barrier()  # synchronizes all processes
            dist.all_reduce(loss, op=torch.distributed.ReduceOp.AVG,)  # get mean loss for all processes

            # ascent
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            if a_iter == cfg.start_sam_epoch:
                momentum_grad = minimizer.initiate_mask_relevant()
            elif a_iter != cfg.start_sam_epoch and a_iter % cfg.update_mask_fre == 1:
                momentum_grad = minimizer.update_mask_and_momentum_grad(global_mask, momentum_grad)
            elif a_iter > cfg.start_sam_epoch and a_iter % cfg.update_mask_fre != 1:
                minimizer.get_history_mask(local_mask)
            minimizer.ascent_step()
            scaler.update()
            del output, loss
                
            # descent
            output = model(data)

            # loss
            if cfg.cal_confidence_weight == True and a_iter > cfg.warmup_epoch:
                loss = compute_loss(cfg, a_iter, loss_fun, output, outputs_teacher_all, confidence_score=confidence_score, confidence_statics=confidence_statics)
            else:
                loss = compute_loss(cfg, a_iter, loss_fun, output, outputs_teacher_all)

            dist.barrier()  # synchronizes all processes
            dist.all_reduce(loss, op=torch.distributed.ReduceOp.AVG,)  # get mean loss for all processes

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            minimizer.descent_step()
            scaler.update()
            
            loss_all = loss_all + loss.item()

            # calculate mertic
            compute_metric(cfg, output, target, post_pred, hausdorff_metric, dice_metric)

        running_loss = running_loss + (loss.item() * cfg.batch_size)
        dataset_size = dataset_size + cfg.batch_size
        
        scheduler.step()

    if a_iter == cfg.start_sam_epoch or a_iter % cfg.update_mask_fre == 1:
        dist.barrier()
        for n, p in model.named_parameters():
            dist.all_reduce(momentum_grad[n], op=torch.distributed.ReduceOp.AVG,)
        local_mask = minimizer.get_local_mask(momentum_grad)
    
    dist.barrier()
    for i in range(cfg.task_num):
        dice.append(dice_metric[i].aggregate().item())
        hausdorff.append(hausdorff_metric[i].aggregate().item())
        dice_metric[i].reset()
        hausdorff_metric[i].reset()

    if cfg.cal_confidence_weight == True:
        confidence_bank[client_idx].aggregate()
        for c in range(cfg.task_num+1):
            classwise_confidence_bank[client_idx][c].update_mean()
            classwise_confidence_bank[client_idx][c].reset()
        
    loss = loss_all / len_data_loader

    model.to('cpu')
    teacher_model_1.to('cpu')
    teacher_model_2.to('cpu')
    teacher_model_3.to('cpu')
    if cfg.global_pseduo == True and a_iter >= cfg.global_pseduo_epoch:
        server_model.to('cpu')
    
    del data, target, loss_all, loss_fun, output, running_loss, outputs_teacher_all
    gc.collect()
    torch.cuda.empty_cache()
    
    return loss, dice, hausdorff, confidence_bank, classwise_confidence_bank, local_mask, momentum_grad