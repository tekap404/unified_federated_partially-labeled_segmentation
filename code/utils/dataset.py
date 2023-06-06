from torch.utils.data import Dataset as _TorchDataset
from monai.transforms import apply_transform
from typing import Sequence, Optional, Callable, Union
from torch.utils.data import Subset
import collections.abc
import torch
import gc
import cupy as cp
from torch.utils.data.distributed import DistributedSampler
from monai.data import DataLoader

class My_Dataset(_TorchDataset):
    def __init__(self, data: Sequence, transform: Optional[Callable] = None, cfg=None, mode=None) -> None:
        self.data = data
        self.transform = transform
        self.cfg = cfg
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        
        data_i_dict = self.data[index]

        img = cp.load(data_i_dict['image'])
        mask = cp.load(data_i_dict['mask'])

        data_i = {}
        # img=(h,w,d), mask=(5,h,w,d)
        data_i['image'], data_i['mask'] = torch.Tensor(img).cuda(self.cfg.device).unsqueeze(0), torch.Tensor(mask).cuda(self.cfg.device)
        data_i['image'], data_i['mask'] = data_i['image'].transpose(1,3), data_i['mask'].transpose(1,3)
        del img, mask
        gc.collect()

        result = {}
        if self.mode == 'train':
            batch = apply_transform(self.transform, data_i)[0]
        else:
            batch = apply_transform(self.transform, data_i)
        if self.mode == 'train':
            result["image"], result["mask"], result["name"] = batch["image"], batch["mask"], data_i_dict['image']
        else:
            result["image"], result["mask"] = batch["image"], batch["mask"]
        return result

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)

        return self._transform(index)

class My_Dataset_test(_TorchDataset):
    def __init__(self, data: Sequence, transform: Optional[Callable] = None, cfg=None, mode=None) -> None:
        self.data = data
        self.transform = transform
        self.cfg = cfg
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        
        data_i_dict = self.data[index]

        img = cp.load(data_i_dict['image'])
        mask = cp.load(data_i_dict['mask'])

        data_i = {}
        data_i['image'], data_i['mask'] = torch.Tensor(img).cuda(self.cfg.device).unsqueeze(0), torch.Tensor(mask).cuda(self.cfg.device)
        data_i['image'], data_i['mask'] = data_i['image'].transpose(1,3), data_i['mask'].transpose(1,3)
        del img, mask
        gc.collect()

        result = {}
        batch = apply_transform(self.transform, data_i)
        result["image"], result["mask"], result["name"] = batch["image"], batch["mask"], data_i_dict['image'].split('/')[-1].split('.')[0]
        return result

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)

        return self._transform(index)


def get_train_dataset(cfg, rand_seed=None):
    train_ds = My_Dataset(data=cfg.data_json["train"], transform=cfg.train_transforms, cfg=cfg, mode='train')

    return train_ds

def get_val_dataset(cfg):
    val_ds = My_Dataset(data=cfg.data_json["val"], transform=cfg.val_transforms, cfg=cfg)
    return val_ds

def get_test_dataset(cfg):
    test_ds = My_Dataset_test(data=cfg.data_json["test"], transform=cfg.test_transforms, cfg=cfg)
    return test_ds

def get_train_dataloader(train_dataset, cfg):

    train_sampler = DistributedSampler(train_dataset, num_replicas=cfg.world_size, rank=cfg.local_rank)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        drop_last=True,
        sampler=train_sampler,
    )
    return train_dataloader

def get_val_dataloader(val_dataset, cfg):

    val_sampler = DistributedSampler(val_dataset, num_replicas=cfg.world_size, rank=cfg.local_rank)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        num_workers=0,
        sampler=val_sampler,
    )
    return val_dataloader