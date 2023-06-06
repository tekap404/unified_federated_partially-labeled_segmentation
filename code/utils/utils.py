import torch
from torch import nn
import numpy as np
import random
import os
import math
from torch.nn import init
from monai.metrics import CumulativeIterationMetric
from .dataset import *
from .loss import *
from .optimizer import *
from .scheduler import *

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True#False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')

def str2bool(str):
	return True if str.lower() == 'true' else False

def get_kernels_strides(cfg):
    sizes, spacings = cfg.img_size, cfg.spacing
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides
    
def mix_all_teacher_preds(logits_teacher, outputs_teacher, i, logits_teacher_all, outputs_teacher_all):

    # accumulation of background channel
    logits_teacher_all[:,0] += logits_teacher[:,0]
    outputs_teacher_all[:,0] += outputs_teacher[:,0]

    # only use pred from the teacher model that is trained on specific channel(s)
    if i == 0:  # kidney
        logits_teacher_all[:,1] += logits_teacher[:,1]
        outputs_teacher_all[:,1] += outputs_teacher[:,1]
    elif i == 1:    # spleen and pancreas
        logits_teacher_all[:,3] += logits_teacher[:,1]
        logits_teacher_all[:,4] += logits_teacher[:,2]
        outputs_teacher_all[:,3] += outputs_teacher[:,1]
        outputs_teacher_all[:,4] += outputs_teacher[:,2]
    elif i == 2:    # lung
        logits_teacher_all[:,2] += logits_teacher[:,1]
        outputs_teacher_all[:,2] += outputs_teacher[:,1]
    return logits_teacher_all, outputs_teacher_all

@torch.no_grad()
def get_teachers_intersection(cfg, outputs_teacher_all, outputs_oralteacher_all):

    # calculate volume of foreground pred from the global model
    pixel_num = torch.sum(outputs_teacher_all[1:].view(-1))
    intersec_T = int(pixel_num * cfg.intersec_tao)
    intersec = torch.tensor(outputs_teacher_all[1:] * outputs_oralteacher_all[1:], dtype=torch.int8)
    intersec_num = int(torch.sum(intersec.view(-1)))

    if intersec_num < intersec_T:
        return outputs_teacher_all
    else:
        return torch.tensor(outputs_teacher_all * outputs_oralteacher_all, dtype=torch.int8)

def normal_distribution(cfg, x, mean, sigma, epoch, T_range):
    if epoch <= cfg.final_weight_epoch//2:
        x_shift_range = T_range / (cfg.final_weight_epoch//2) * epoch
    else:
        x_shift_range = - T_range / (cfg.final_weight_epoch//2) * (epoch - cfg.final_weight_epoch//2) + T_range
    p_x = torch.exp( -((x-mean-x_shift_range)**2) / (2*(sigma**2)) )/(math.sqrt(2*np.pi) * sigma)
    return p_x

def get_confidence(pred, pred_post):
    pred = nn.Softmax()(pred)
    C= pred.size()[0]
    confidence = 0
    count = 0
    for c in range(C):
        pixel_confidence = -torch.sum(torch.mul(pred, torch.log(pred + 1e-7)), dim=0) / C
        confidence_c = torch.sum(pixel_confidence * pred_post[c]) / (torch.sum((pred_post[c].view(-1))) + 1)
        if confidence_c != 0:
            confidence += confidence_c
            count += 1
    if count != 0:
        confidence /= count  
    else:
        confidence = torch.zeros((1))
    
    return confidence

def get_classwise_confidence(pred, pred_post):
    pred = nn.Softmax()(pred.unsqueeze(0)).squeeze(0)
    pixel_confidence = - torch.mul(pred, torch.log(pred + 1e-7))
    confidence_c = torch.sum(pixel_confidence * pred_post) / (torch.sum((pred_post.view(-1))) + 1)
    
    return confidence_c

def get_confidence_weight(cfg, confidence_score, confidence_statics, a_iter):

    confidence_mean, confidence_max, confidence_min, T = confidence_statics[0].item(), confidence_statics[1].item(), confidence_statics[2].item(), confidence_statics[3].item()
    confidence_mean, confidence_max, confidence_min = confidence_statics[0].item(), confidence_statics[1].item(), confidence_statics[2].item()
    weight = torch.tensor(confidence_score).to(confidence_score.device)
    
    for n in range(len(confidence_score)):
        # # RG
        # cur = (confidence_score[n] - confidence_mean) / (confidence_max - confidence_min)
        # min = (confidence_min - confidence_mean) / (confidence_max - confidence_min)
        # T_range = (T - confidence_mean) / (confidence_max - confidence_min) - (confidence_min - confidence_mean) / (confidence_max - confidence_min)
        # weight[n] = normal_distribution(cfg, cur, min, cfg.sigma, a_iter, T_range)
        # weight[n] = (1 - cfg.weight_k) * weight[n] + cfg.weight_k * ( 2 - torch.exp( (confidence_score[n] - confidence_mean) / (confidence_max - confidence_min) ) )
        
        # TS
        if confidence_score[n] > T:
            weight[n] =  2 - torch.exp( (confidence_score[n] - confidence_mean) / (confidence_max - confidence_min) - (a_iter / cfg.final_weight_epoch) )
        else:
            weight[n] =  2 - torch.exp( (confidence_score[n] - confidence_mean) / (confidence_max - confidence_min) )
           
        # # BD
        # weight[n] =  2 - torch.pow((cfg.base_alpha-math.exp(1)) / cfg.final_weight_epoch * a_iter + math.exp(1), (confidence_score[n] - confidence_mean) / (confidence_max - confidence_min) )

    return weight

class Confidence_bank(CumulativeIterationMetric):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.mean = 0

    def push(self, confidence_score):
        self.append(confidence_score)

    def _compute_tensor(self, pred, gt):
        return
    
    def aggregate(self):
        data = self.get_buffer()
        return data

    def update_mean(self):
        data = self.get_buffer()
        mean = torch.mean(data)
        if self.mean == 0:
            self.mean = mean
        else:
            self.mean = self.cfg.mom*self.mean + (1-self.cfg.mom)*mean

    def get_statics(self):
        data = self.get_buffer()
        mean = torch.mean(data)
        max = torch.max(data)
        min = torch.min(data)
        T = torch.quantile(data, self.cfg.confidence_threshold)
        return (mean, max, min, T)

def update_client_weight_by_confidence(cfg, num_weights, confidence_bank):

    confidence_mean_weight = [0 for _ in range(cfg.client_num)]
    confidence_var_weight = [0 for _ in range(cfg.client_num)]
    confidence_list = []
    for i in range(len(confidence_bank)):
        confidence_list.append(confidence_bank[i].aggregate())

    # calculate aggregation weight for statistical mean
    mean_all = 0
    for i in range(cfg.client_num):
        mean_all += torch.exp( (- torch.mean(confidence_list[i])) / cfg.confidence_mean_t )
    for i in range(cfg.client_num):
        confidence_mean_weight[i] = (torch.exp( (- torch.mean(confidence_list[i])) / cfg.confidence_mean_t )) / mean_all

    # calculate aggregation weight for statistical var
    var_all = 0
    for i in range(cfg.client_num):
        var_all += torch.exp( ( torch.var(confidence_list[i])) / cfg.confidence_var_t )
    for i in range(cfg.client_num):
        confidence_var_weight[i] = (torch.exp( ( torch.var(confidence_list[i])) / cfg.confidence_var_t )) / var_all

    client_weights = []
    for a,b,c in zip(num_weights, confidence_mean_weight, confidence_var_weight):
        client_weights.append( (a+b+c)/3 )

    return client_weights

def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint

def init_weights(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>