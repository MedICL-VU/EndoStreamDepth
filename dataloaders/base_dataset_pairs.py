import os
import cv2
import torch
import numpy as np
import logging
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from PIL import Image
import h5py
import torch.distributed as dist
import OpenEXR
import pickle
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class BaseDatasetPairs(Dataset):
    @classmethod
    def create(cls, dataset_name, root_dir=None, split='train', load_cache=None, disparity=False):
        if dataset_name.lower() == 'cv3d':
            from .cv3d_dataset import Cv3dDepth as DepthDataset
        elif dataset_name.lower() == 'cv3d2':
            from .cv3d2_dataset import Cv3d2Depth as DepthDataset
        elif dataset_name.lower() == 'simcol3d':
            from .simcol3d import SimCol3DDepth as DepthDataset

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return DepthDataset(root_dir=root_dir, split=split, load_cache=load_cache, disparity=disparity)
        

    def __init__(self, dataset_name, root_dir, split, load_cache=None):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.split = split
        self.reshape_list = {'resolution': (), 'crop_type': None, 'stride': 1, 'resize_factor': 1}
        self._initialize_pairs(load_cache)

    def _initialize_pairs(self, load_cache):
        if load_cache is not None and os.path.exists(self.get_cache_path(load_cache)):
            logging.info(f"loading cache for {self.dataset_name}")
            self.pairs = pickle.load(open(self.get_cache_path(load_cache), 'rb'))
        else:
            self.scenes = []
            self.pairs = []
            self._build_pairs()
            
            if load_cache is not None:
                logging.info(f"saving cache for {self.dataset_name}")
                pickle.dump(self.pairs, open(self.get_cache_path(load_cache), 'wb'))

    def _build_pairs(self):
        scenes_path = self.get_scenes_path()
        all_scenes = self.get_all_scenes(scenes_path)
        filter_scenes = self.get_filter_scenes(self.split)

        for item in all_scenes:
            if self.split != 'train':
                scene_pairs = []
            if item in filter_scenes:
                continue
                
            rgb_path, depth_path = self.get_rgb_depth_paths(scenes_path, item)
            if not self.is_valid_scene(rgb_path, depth_path):
                continue

            self.scenes.append(item)
            all_imgs = self.get_sorted_image_files(rgb_path)
            
            count = 0
            for img_name in all_imgs:
                depth_name = self.get_depth_name(img_name)
                pair_dict = self.create_pair_dict(rgb_path, depth_path, img_name, depth_name, 
                                                count, len(all_imgs), item)
                
                if os.path.exists(os.path.join(depth_path, depth_name)) and self.split == 'train':
                    self.pairs.append(pair_dict)
                    count += 1
                else:
                    scene_pairs.append(pair_dict)
                    
            if self.split != 'train':
                self.pairs.append(scene_pairs)

        if len(self.scenes) == 0:
            raise RuntimeError(f"No valid scenes found in {scenes_path}")

        self.scenes = sorted(self.scenes)
        logging.info(f'number of scenes / length of {self.dataset_name}: {len(self.scenes)} / {len(self.pairs)}')

    # Methods to be overridden by child classes
    def get_cache_path(self, cache_dir):
        raise NotImplementedError

    def get_scenes_path(self):
        return self.root_dir

    def get_all_scenes(self, scenes_path):
        return sorted([s for s in os.listdir(scenes_path) if os.path.isdir(os.path.join(scenes_path, s))])

    def get_filter_scenes(self, split):
        raise NotImplementedError
    
    def get_rgb_depth_paths(self, scenes_path, scene_name):
        raise NotImplementedError

    def is_valid_scene(self, rgb_path, depth_path):
        return (os.path.isdir(rgb_path) and 
                (os.path.isdir(depth_path) or self.split != 'train'))

    def get_sorted_image_files(self, rgb_path):
        raise NotImplementedError

    def get_depth_name(self, img_name):
        raise NotImplementedError

    def create_pair_dict(self, rgb_path, depth_path, img_name, depth_name, 
                        scene_index, scene_length, scene_name):
        return {
            'image': os.path.join(rgb_path, img_name),
            'depth': os.path.join(depth_path, depth_name),
            'scene_index': scene_index,
            'scene_length': scene_length,
            'scene_name': scene_name
        }

    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):

        pass

    def depth_read(self, path):
        raise NotImplementedError