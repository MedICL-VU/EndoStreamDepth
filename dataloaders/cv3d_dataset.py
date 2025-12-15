import os
import cv2
import torch
import numpy as np
import logging
from .base_dataset_pairs import BaseDatasetPairs
import tifffile


testing_scenes = ['trans_t1_a', 'trans_t1_b', 'trans_t2_a', 'trans_t2_b', 'trans_t2_c', 'trans_t3_a', 'trans_t3_b', 'trans_t4_a', 'trans_t4_b']
class Cv3dDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None, disparity=False):
        self.root_dir = os.path.join(root_dir, 'cv3d')
        super().__init__(dataset_name='cv3d', root_dir=self.root_dir, split=split, load_cache=load_cache)
        self.reshape_list['resolution'] = (518, 518) # 1350, 1078
        self.reshape_list['stride'] = 1
        self.disparity = disparity

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, f'cv3d_pairs_{self.split}.pkl')

    def get_all_scenes(self, scenes_path):
        all_scenes = [s for s in os.listdir(scenes_path)
                      if os.path.isdir(os.path.join(scenes_path, s))]
        return sorted(all_scenes)


    def get_filter_scenes(self, split):
        all_scenes = self.get_all_scenes(self.get_scenes_path())
        if split == 'val':
            return [s for s in all_scenes if s not in ['sigmoid_t3_a', 'sigmoid_t3_b']]  # only use these two scenes
        elif split == 'train':
            return ['sigmoid_t3_a', 'sigmoid_t3_b'] + testing_scenes  # leave for validation
        elif split == 'test':
            return [s for s in all_scenes if s not in testing_scenes] # only use the 30 testing scenes
        return []

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        item_path = os.path.join(scenes_path, scene_name)
        return (item_path,
                item_path)

    def get_sorted_image_files(self, rgb_path):
        all_imgs = [f for f in os.listdir(rgb_path) if f.endswith('_color.png')]
        return sorted(all_imgs, key=lambda x: int(os.path.basename(x).split('_color.png')[0]))

    def get_depth_name(self, img_name):
        return img_name.replace('_color.png', '_depth.tiff')

    def depth_read(self, path, return_torch=False, **kwargs):
        depth = tifffile.imread(path).astype(float)
        # TODO: Depth frame: depth along the camera frame’s z-axis, clamped from 0-100 millimeters. Values are linearly scaled and encoded as a 16-bit grayscale image.
        depth = depth / 65535 * 100


        inverse_depth = np.zeros_like(depth)
        valid_mask = depth > 0
        if self.disparity:
            inverse_depth[valid_mask] = 1000.0 / depth[valid_mask] # 1/scale
            print('using disparity gt')
        else:
            inverse_depth[valid_mask] = depth[valid_mask]

        if kwargs.get('print_minmax', False):
            logging.info(f"minmax depth for {path}: {inverse_depth.min():.3f}, {inverse_depth.max():.3f}")

        if return_torch:
            inverse_depth = torch.from_numpy(inverse_depth).float()
        return inverse_depth