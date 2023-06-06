import os
from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# data path
cfg.data_dir = "../data/"
cfg.output_dir = "../output/"

# dataset
cfg.batch_size = 4
cfg.val_batch_size = 4
cfg.img_size = (160, 160, 64)

# training
cfg.lr = 1e-3
cfg.min_lr = 1e-5
cfg.weight_decay = 0
cfg.epochs = 500
cfg.seed = 0
cfg.eval_epochs = 1
cfg.start_cal_metric_epoch = 1
cfg.warmup = 1

basic_cfg = cfg
