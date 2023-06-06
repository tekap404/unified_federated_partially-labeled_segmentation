import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    NormalizeIntensityd,
    RandFlipd,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCoarseDropoutd,
    RandGridDistortiond,
    OneOf,
    RandCropByPosNegLabeld,
    SpatialPadd,
    RandRotate90d,
)
from default_config import basic_cfg

cfg = basic_cfg

# train
cfg.train = True
cfg.eval = True
cfg.base_num_features = 32

# lr
cfg.lr_mode = "warmup_restart"
cfg.lr = 1e-4
cfg.min_lr = 1e-6
cfg.weight_decay = 1e-6
cfg.epochs = 501
cfg.restart_epoch = 3           # only for warmup_restart
cfg.multiplier = 1
cfg.warmup_epoch = 10

# dataset
cfg.img_size = (80, 192, 192)
cfg.batch_size = 2  
cfg.val_batch_size = 1

# val
cfg.eval_epochs = 5             # eval frequency
cfg.start_eval_epoch = 10       # when using large lr, one can set a large num
cfg.roi_size = (80, 192, 192)
cfg.sw_batch_size = 2

# FL
cfg.local_round = 1
cfg.com_epoch = 0               # start_epoch
cfg.com_fre = 1                 # communication frequency
cfg.client_num = 3
cfg.task_num = 4                # 1/2 for SOLO, 4 for others

# pseudo label framework
# aRCE loss
cfg.rce = True
cfg.rce_alpha = 0.01

cfg.cal_confidence_weight = False

# weight loss according to uncertainty
cfg.confidence_weight_loss = True
cfg.mom = 0.9                           # momentum to update mean of uncertainty
cfg.final_weight_epoch = 200
cfg.confidence_threshold = 0.7
cfg.confidence_threshold_max = 0.3      # TS
cfg.base_alpha = 1                      # BD
cfg.sigma = 0.28                        # RG
cfg.weight_k = 0.7                      # RG
if cfg.confidence_weight_loss == True:
    cfg.cal_confidence_weight = True

# change agg weight according to mean and variance of uncertainty bank
cfg.confidence_change_com = True
cfg.ua_epoch = 300
cfg.confidence_mean_t = 0.05
cfg.confidence_var_t = 1e-3
cfg.confidence_change_part = ['decoder'] # ['decoder'] / ['encoder', 'supervision'] / ['encoder', 'supervision', 'decoder']
if cfg.confidence_change_com == True:
    cfg.cal_confidence_weight = True

# GMT
cfg.global_pseduo = True
cfg.global_pseduo_epoch = 300           # the starting epoch of GMT
cfg.global_intersec = True
cfg.intersec_tao = 0.8

# sUSAM
cfg.sam = True
cfg.start_sam_epoch = 300
cfg.rho = 0.7    
cfg.eta = 1e-3
cfg.sparse = True
cfg.topkgrad = 0.4
cfg.momentum_gamma = 0.9
cfg.nonintersec_p = 0.1
cfg.update_mask_fre = 5

# GIN-IPA (CMIDG)
cfg.blend_epsilon = 0.3
cfg.blend_grid_size = 24
cfg.gin_nlayer = 4
cfg.gin_n_interm_ch = 1

# path of pretrained teacher weights
cfg.pretrained_teacher_path = [
                 '../output/seed20220930/solo_client1_part123/best_weights.pth',
                 '../output/seed20220930/solo_client2_part123/best_weights.pth',
                 '../output/seed20220930/solo_client3_part123/best_weights.pth',
]

# model
cfg.output_dir = "../output/"

# resume
cfg.resume = False
cfg.weights_path = '../output/seed20220930/UFPS_part123/latest_weights.pth'

# test
cfg.test_path = '../output/seed20220930/solo_client1_part123/best_weights.pth'

# transforms
cfg.train_transforms = Compose(
    [
        NormalizeIntensityd(keys=("image"), subtrahend=0, divisor=500),       # normalize to [-1,1]
        SpatialPadd(keys=("image", "mask"), spatial_size=cfg.img_size),
        RandCropByPosNegLabeld(     # since background>0, pos and neg make no use, but image_threshold works
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=cfg.img_size,
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        RandRotate90d(keys=("image", "mask"), prob=0.5, spatial_axes=(1, 2)),
        RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[2]),
        RandAffined(
            keys=("image", "mask"),
            prob=0.5,
            rotate_range=np.pi / 12,
            translate_range=(cfg.img_size[1]*0.0625, cfg.img_size[2]*0.0625),
            scale_range=(0.1, 0.1),
            mode="nearest",
            padding_mode="reflection",
        ),
        OneOf(
            [
                RandGridDistortiond(keys=("image", "mask"), prob=0.5, distort_limit=(-0.05, 0.05), mode="nearest", padding_mode="reflection"),
                RandCoarseDropoutd(
                    keys=("image", "mask"),
                    holes=5,
                    max_holes=8,
                    spatial_size=(1, 1, 1),
                    max_spatial_size=(6, 12, 12),
                    fill_value=0.0,
                    prob=0.5,
                ),
            ]
        ),
        RandScaleIntensityd(keys="image", factors=(-0.2, 0.2), prob=0.3),
        RandShiftIntensityd(keys="image", offsets=(-0.1, 0.1), prob=0.3),
        EnsureTyped(keys=("image", "mask"), dtype=torch.float32),
    ]
)

cfg.val_transforms = Compose(
    [
        NormalizeIntensityd(keys=("image"), subtrahend=0, divisor=500),
        SpatialPadd(keys=("image", "mask"), spatial_size=cfg.roi_size),
        EnsureTyped(keys=("image", "mask"), dtype=torch.float32),
    ]
)

cfg.test_transforms = Compose(
    [
        NormalizeIntensityd(keys=("image"), subtrahend=0, divisor=500),
        SpatialPadd(keys=("image", "mask"), spatial_size=cfg.roi_size),
        EnsureTyped(keys=("image", "mask"), dtype=torch.float32),
    ]
)