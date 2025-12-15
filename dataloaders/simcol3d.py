import os
import cv2
import torch
import numpy as np
import logging
from .base_dataset_pairs import BaseDatasetPairs
import tifffile, imageio


testing_scenes = ['Frames_S5', 'Frames_S10', 'Frames_S15', 'Frames_B5', 'Frames_B10', 'Frames_B15', 'Frames_O1', 'Frames_O2',
                  'Frames_O3']
class SimCol3DDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None, disparity=False):
        self.root_dir = os.path.join(root_dir, 'simcol3d_all_in_one')
        super().__init__(dataset_name='simcol3d', root_dir=self.root_dir, split=split, load_cache=load_cache)
        self.reshape_list['resolution'] = (476, 476) # 1350, 1078
        self.reshape_list['stride'] = 1
        self.disparity = disparity

    def get_cache_path(self, cache_dir):
        return os.path.join(cache_dir, f'simcol3d_pairs_{self.split}.pkl')

    def get_all_scenes(self, scenes_path):
        all_scenes = [s for s in os.listdir(scenes_path)
                      if os.path.isdir(os.path.join(scenes_path, s))]
        return sorted(all_scenes)


    def get_filter_scenes(self, split):
        all_scenes = self.get_all_scenes(self.get_scenes_path())
        if split == 'val':
            return testing_scenes
        elif split == 'train':
            return testing_scenes
        elif split == 'test':
            return [s for s in all_scenes if s not in testing_scenes]
        return []

    def get_rgb_depth_paths(self, scenes_path, scene_name):
        item_path = os.path.join(scenes_path, scene_name)
        return (item_path,
                item_path)

    def get_sorted_image_files(self, rgb_path):
        # Keep only FrameBuffer_XXXX.png files
        all_imgs = [
            f for f in os.listdir(rgb_path)
            if f.startswith("FrameBuffer_") and f.endswith(".png")
        ]

        # Extract the numeric part after "FrameBuffer_"
        def frame_index(fname: str) -> int:
            stem = os.path.splitext(os.path.basename(fname))[0]  # "FrameBuffer_0000"
            return int(stem.split("FrameBuffer_")[-1])  # "0000" -> 0

        return sorted(all_imgs, key=frame_index)

    def get_depth_name(self, img_name):
        return img_name.replace('FrameBuffer', 'Depth')

    def depth_read(self, path, return_torch=False, **kwargs):
        depth = imageio.imread(path)
        depth = depth / 65280 # https://github.com/anitarau/simcol/blob/main/evaluation/eval_synthetic_depth.py#L31


        inverse_depth = np.zeros_like(depth)
        valid_mask = depth > 0
        if self.disparity:
            inverse_depth[valid_mask] = 1000.0 / depth[valid_mask] # 1/scale
        else:
            inverse_depth[valid_mask] = depth[valid_mask]

        if kwargs.get('print_minmax', False):
            logging.info(f"minmax depth for {path}: {inverse_depth.min():.3f}, {inverse_depth.max():.3f}")

        if return_torch:
            inverse_depth = torch.from_numpy(inverse_depth).float()

        return inverse_depth