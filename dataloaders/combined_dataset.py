import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torch.distributed as dist
import numpy as np
import logging
import os
from os.path import join
import math
import cv2
from .depthanything_preprocess import _load_and_process_image, _load_and_process_depth
from .base_dataset_pairs import BaseDatasetPairs
import albumentations as A
from albumentations.core.composition import Compose
from albumentations import ReplayCompose
import random

import PIL.Image as Image

def compute_vit_crop(h, w, patch=14):
    new_h = (h // patch) * patch
    new_w = (w // patch) * patch
    return new_h, new_w

class ResizeWithSeparateMaskModes(A.Resize):
    def apply(self, img, interpolation=cv2.INTER_AREA, **params):
        # this is used for image
        return cv2.resize(img, (self.width, self.height))

    def apply_to_mask(self, mask, interpolation=cv2.INTER_NEAREST, **params):
        target = params.get("target")
        if target == "valid_mask":
            # valid_mask → nearest neighbor
            return cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        else:
            # depth mask → area
            return cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_AREA)# INTER_AREA

    def get_params_dependent_on_targets(self, params):
        return {}

    @property
    def targets_as_params(self):
        return ["image", "mask", "valid_mask"]



class CombinedDataset(Dataset):
    def __init__(self, root_dir, enable_dataset_flags, resolution=None, split='train',
                 video_length=8, seed=42, tmp_res=None, color_aug=False, disparity=False):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        cache_dir = '/home/hao/hao/EndoStreamDepth/dataloaders/pairs_cache' if split != 'test' else None

        self.pairslist = {}
        self.depth_read_list = {}
        self.reshape_list = {}
        self.tmp_res = tmp_res


        for dataset_name in enable_dataset_flags:
            dataset = BaseDatasetPairs.create(dataset_name, root_dir, split, load_cache=cache_dir, disparity=disparity)
            self.pairslist[dataset_name] = dataset.pairs
            self.depth_read_list[dataset_name] = dataset.depth_read
            self.reshape_list[dataset_name] = dataset.reshape_list

        if resolution == 'base':
            for dataset in self.reshape_list:
                if dataset in ['simcol3d']:
                    self.reshape_list[dataset]['resolution'] = (476, 476) # 518, 518
                elif dataset in ['cv3d', 'cv3d2']:
                    self.reshape_list[dataset]['resolution'] = (518, 518)  # 518, 518

        self.pairs = []

        for dataset_name in enable_dataset_flags:
            indices = list(range(len(self.pairslist[dataset_name])))
            self.pairs.extend([(dataset_name, i) for i in indices])
            logging.info(f"length of {dataset_name} for {split}: {len(self.pairslist[dataset_name])}")

        if split != 'train':
            logging.info(f"enabled datasets for {split}: {enable_dataset_flags}")
            logging.info(f"length of combined dataset: {len(self.pairs)}")


        self.video_length = video_length
        self.split = split

        self.transforms = {}
        """SimCol3D"""
        # self.transformation = ReplayCompose([
        #     # A.Resize(518, 518, mask_interpolation=cv2.INTER_AREA),
        #     A.Resize(476, 476, mask_interpolation=cv2.INTER_AREA),
        #     A.RandomRotate90(),
        #     A.HorizontalFlip(p=0.5),
        #     A.VerticalFlip(p=0.5),
        #     # A.Affine((0.7, 1.3), {'x': (-0.3, 0.3), 'y': (-0.3, 0.3)}, rotate=(-360, 360), p=0.25),
        #     A.GaussianBlur(p=0.1),
        #     A.AutoContrast(p=0.1),
        #     # A.MotionBlur(p=0.1),
        #     A.MedianBlur(blur_limit=15, p=0.1),
        #     A.RandomGamma(p=0.1),
        #     A.Defocus(p=0.1),
        #     A.RandomFog(alpha_coef=0.1, p=0.1),
        #     A.RandomBrightnessContrast(p=0.1),
        #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     A.ToTensorV2()
        # ], additional_targets={'valid_mask': 'mask'})
        #
        # # raw_h = 1080
        # # raw_w = 1350
        # # crop_h, crop_w = compute_vit_crop(raw_h, raw_w, patch=14)
        # #

        """C3VD"""
        self.transformation = ReplayCompose([
            ResizeWithSeparateMaskModes(518, 518),
            # A.CenterCrop(height=crop_h, width=crop_w, p=1.0),

            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            A.GaussianBlur(p=0.1),
            A.AutoContrast(p=0.1),
            A.MotionBlur(p=0.1),
            A.MedianBlur(blur_limit=15, p=0.1),
            A.RandomGamma(p=0.1),
            A.Defocus(p=0.1),
            A.RandomFog(alpha_coef=0.1, p=0.1),
            A.RandomBrightnessContrast(p=0.1),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2()
        ], additional_targets={
            'mask': 'mask',  # your depth – gets INTER_AREA
            'valid_mask': 'mask'  # but overridden to INTER_NEAREST in custom resize
        })


    def _get_transform(self, dataset_name: str):
        size = tuple(self.reshape_list[dataset_name]['resolution'])
        key = (dataset_name, size) # ('simcol3d', (476, 476))
        if key not in self.transforms:
            self.transforms[key] = self._build_transform_for(dataset_name, size)
        return self.transforms[key]


    def _build_transform_for(self, dataset_name: str, size_hw):
        H, W = map(int, size_hw)
        if dataset_name.lower()in ['cv3d', 'cv3d2']:
            resize_stage = ResizeWithSeparateMaskModes(H, W)
            #resize_stage = A.Resize(H, W)
        elif dataset_name.lower() in ['simcol3d'] :
            resize_stage = A.Resize(H, W, mask_interpolation=cv2.INTER_AREA)
        logging.info(f'preprocessing dataset {dataset_name} with the shape {size_hw}')
        return ReplayCompose([
            resize_stage,
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(p=0.1),
            A.AutoContrast(p=0.1),
            A.MedianBlur(blur_limit=15, p=0.1),
            A.RandomGamma(p=0.1),
            A.Defocus(p=0.1),
            A.RandomFog(alpha_coef=0.1, p=0.1),
            A.RandomBrightnessContrast(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2()
        ], additional_targets={
            'mask': 'mask',
            'valid_mask': 'mask'
        })




    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        if self.split == 'val':
            dataset_idx, scene_idx = self.pairs[idx]
            scene = self.pairslist[dataset_idx][scene_idx]

            images = []
            depths = []
            valid_masks = []
            for pair in scene:
                image, _current_crop = _load_and_process_image(pair['image'], **self.reshape_list[dataset_idx])
                depth = self.depth_read_list[dataset_idx](pair['depth']) # needed for scaling focal length, currently only for Spring
                valid_mask = (depth > 0).astype(np.float32)
                depth = _load_and_process_depth(depth, image.shape, _current_crop, **self.reshape_list[dataset_idx])
                valid_mask = cv2.resize(valid_mask, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST)
                # image = image * valid_mask[None, :, :]

                images.append(image)
                depths.append(depth)
                valid_masks.append(torch.from_numpy(valid_mask).float())
                #depths.append(torch.from_numpy(depth).float()) # not resizing depth, using original resolution

            return_name = dataset_idx
            return torch.stack(images).float(), torch.stack(depths).float(), torch.stack(valid_masks).float(), return_name


        elif self.split == 'test':
            dataset_idx, scene_idx = self.pairs[idx]
            scene = self.pairslist[dataset_idx][scene_idx]

            images = []
            depths = []
            for pair in scene:
                image, _current_crop = _load_and_process_image(pair['image'], **self.reshape_list[dataset_idx])
                depth = self.depth_read_list[dataset_idx](pair['depth']) # needed for scaling focal length, currently only for Spring
                depth = _load_and_process_depth(depth, image.shape, _current_crop, **self.reshape_list[dataset_idx])
                depths.append(depth) # not resizing depth, using original resolution
                images.append(image)
                # depths.append(torch.from_numpy(depth).float()) # not resizing depth, using original resolution

            return_name = os.path.join(dataset_idx, pair['scene_name'])
            return torch.stack(images).float(), torch.stack(depths).float(), return_name

        # dataset_idx: i-th dataset; e.g. pointodyssey is 0, spring is 1...etc
        # pair_idx: the i-th (img, depth) pair in the dataset, for instance, pair_idx \in [0, 5000] in Spring
        dataset_idx, pair_idx = self.pairs[idx]
        dataset_list = self.pairslist[dataset_idx]
        pair = dataset_list[pair_idx]


        scene_index = pair['scene_index']
        scene_length = pair['scene_length']
        stride = self.reshape_list[dataset_idx]['stride']


        # Check if we can go both forward and backward
        can_go_forward = scene_index + (self.video_length - 1) * stride <= scene_length - 1
        can_go_backward = scene_index >= (self.video_length - 1) * stride

        if can_go_forward and can_go_backward:
            # Randomly choose direction
            if torch.rand(1).item() > 0.5:
                sequence_indices = list(range(scene_index, scene_index + self.video_length * stride, stride))
            else:
                start_pos = scene_index - (self.video_length - 1) * stride
                sequence_indices = list(range(start_pos, scene_index + 1, stride))
        elif can_go_forward:
            # Only enough frames ahead
            sequence_indices = list(range(scene_index, scene_index + self.video_length * stride, stride))
        elif can_go_backward:
            # Must go backward
            start_pos = scene_index - (self.video_length - 1) * stride
            sequence_indices = list(range(start_pos, scene_index + 1, stride))
        else:
            # Can't go either way - use remaining frames forward then wrap around backward
            remaining_forward = scene_length - scene_index
            remaining_forward_frames = math.ceil(remaining_forward / stride)
            remaining_needed = max(self.video_length - remaining_forward_frames, 0)

            # Get forward frames
            sequence_indices = list(range(scene_index, scene_length, stride))

            # Add backward frames if needed
            if remaining_needed > 0:
                start = scene_index - remaining_needed * stride
                backward_indices = list(range(start, scene_index, stride))
                sequence_indices.extend(backward_indices)

            # Final safeguard to enforce video_length
            if len(sequence_indices) > self.video_length:
                sequence_indices = sequence_indices[:self.video_length]
            elif len(sequence_indices) < self.video_length:
                # repeat the last frame
                sequence_indices.append(sequence_indices[-1])

        # Get the base offset for this scene in the flat list
        scene_start_idx = pair_idx - scene_index  # This gives us the index where this scene starts



        # Load all frames in sequence
        images = []
        depths = []
        valid_masks = []
        # Transform scene-relative indices to dataset-relative indices

        # transform = self._get_transform(dataset_idx)
        replay = None

        sequence_indices = [scene_start_idx + s for s in sequence_indices]
        for seq_i, seq_idx in enumerate(sequence_indices):
            try:
                pair = dataset_list[seq_idx]
            except Exception as e:
                print("dataset, pair idx: ", dataset_idx, pair_idx)
                print(f"seq indices: {sequence_indices}")
                print("pairslist len: ", len(self.pairslist[dataset_idx]))
                dist.barrier()

            image = np.array(Image.open(pair['image']).convert("RGB"))
            depth = self.depth_read_list[dataset_idx](pair['depth'])
            valid_mask = (depth > 0).astype(np.float32)

            if seq_i == 0:
                # sample random augmentation once for the whole sequence
                # augmented = transform(image=image, mask=depth, valid_mask=valid_mask)
                augmented = self.transformation(image=image, mask=depth, valid_mask=valid_mask)
                replay = augmented["replay"]
            else:
                # replay exactly the same augmentation params
                augmented = A.ReplayCompose.replay(
                    replay,
                    image=image,
                    mask=depth,
                    valid_mask=valid_mask,
                )

            image = augmented['image']
            depth = augmented['mask']
            valid_mask = augmented["valid_mask"]

            # valid_mask = cv2.resize(valid_mask, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST)


            # import matplotlib.pyplot as plt
            # plt.imshow(image.permute(1, 2, 0))
            # plt.title(f"{pair['image']}")
            # # plt.imshow(depth)
            # plt.title(f"{pair['depth']}")
            #
            # # plt.imshow(np.moveaxis(image, 0,2))
            # plt.show()
            # # print(np.unique(image))
            # print(1)


            images.append(image)
            depths.append(depth)
            # valid_masks.append(torch.from_numpy(valid_mask).float())
            valid_masks.append(valid_mask)

        try:
            images = torch.stack(images, dim=0)  # [T, C, H, W]
            depths = torch.stack(depths, dim=0) if self.split != 'test' else None  # [T, H, W]
            valid_masks = torch.stack(valid_masks, dim=0) if self.split != 'test' else None

        except Exception as e:
            import ipdb; ipdb.set_trace()
            dist.barrier()

        return images.float(), depths, valid_masks, dataset_idx #, pair['scene_name']
