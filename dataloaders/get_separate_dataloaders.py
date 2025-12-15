import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from dataloaders.combined_dataset import CombinedDataset
import logging

def get_dataloader(cfg, dataset_name, len_dataloaders=1):
        
    dataset = CombinedDataset(root_dir=cfg.dataset.data_root, resolution=cfg.dataset.resolution, enable_dataset_flags=dataset_name, video_length=cfg.dataset.video_length, split='train')
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, num_workers=int(cfg.training.workers/len_dataloaders), pin_memory=False,
                                  drop_last=True, sampler=sampler) # drop_last=True because current mamba setup seems to require fixed sequence length
    return train_dataloader


def get_all_dataloaders(cfg):
    dataloaders = []
    len_dataloaders = len(cfg.dataset.train_datasets)
    for dataset in cfg.dataset.train_datasets:
        dataloaders.append(get_dataloader(cfg, [dataset], len_dataloaders))
    return dataloaders