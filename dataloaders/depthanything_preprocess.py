import cv2
import numpy as np
from torchvision import transforms as T 
import torch
from torchvision.transforms import functional as F


def depthanything_preprocess(image, width=None, height=None, to_tensor=True, color_aug=False):
    
    width = width or image.shape[1]
    height = height or image.shape[0]

    jitter_params = dict(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
    color_jitter  = T.ColorJitter(**jitter_params)

    def _maybe_jitter(sample):
        to_chw  = lambda x: torch.from_numpy(x).permute(2, 0, 1)        # HWC → CHW
        to_hwc  = lambda x: x.permute(1, 2, 0).numpy()                  # CHW → HWC

        """Apply jitter to sample['image'] (NumPy HWC in [0,1])"""
        if not color_aug:
            # print('nonononono')
            return sample                     # no-op
        img = sample["image"]
        # convert → tensor → jitter → back to NumPy so the rest of the pipeline is unchanged.
        img_t = to_chw(img)                   # CHW, float32, [0,1]
        img_t = color_jitter(img_t)           # apply in PyTorch space
        sample["image"] = to_hwc(img_t)       # back to HWC
        return sample


    transform = T.Compose([
            Resize(
                width=width,
                height=height,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            lambda x: {'image': np.clip(x['image'], 0, 1)},
            _maybe_jitter,
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    image = transform({'image': image})['image']
    if to_tensor:
        image = torch.from_numpy(image).permute(2, 0, 1)
    return image


def _load_and_process_image(image_path, crop_type=None, resolution=None, resize_factor=1.0, **kwargs):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    
    h, w = image.shape[:2]
    _current_crop = None

    if crop_type is not None and resolution is not None:
        target_w, target_h = resolution  # Unpack width and height

        new_h, new_w = int(h * resize_factor), int(w * resize_factor)
        image = cv2.resize(image, (new_w, new_h))
        image = np.clip(image, 0, 1)
        h, w = new_h, new_w
        
        if crop_type == 'center':
            # First resize 
            scale = max(target_h / h, target_w / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            image = np.clip(image, 0, 1)
            
            # Then center crop to target_w x target_h
            start_y = (new_h - target_h) // 2
            start_x = (new_w - target_w) // 2
            image = image[start_y:start_y + target_h, start_x:start_x + target_w]
            _current_crop = (start_y, start_x, target_h, target_w)
        
        elif crop_type == 'random':
            # First resize the image by resize_factor 
            # resize factor here is just a magic number to more easily get a desired crop size relative to the original image
            # e.g. if we have a crop of 512x512 but the image is 4k, we might have too many non-overlapping crops
            # but generally set to 1.0 (i.e. image not resized)
            # new_h, new_w = int(h * resize_factor), int(w * resize_factor)
            # image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            # image = np.clip(image, 0, 1)
            # h, w = new_h, new_w
            
            # ensure it's at least target_w x target_h
            scale = max(target_h / h, target_w / w)
            if scale > 1:
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h))
                image = np.clip(image, 0, 1)
                h, w = new_h, new_w
            
            # Generate random crop coordinates
            max_y = h - target_h
            max_x = w - target_w
            _current_crop = {
                'y': np.random.randint(0, max(1, max_y + 1)),
                'x': np.random.randint(0, max(1, max_x + 1))
            }
        
            # Apply crop
            y, x = _current_crop['y'], _current_crop['x']
            image = image[y:y + target_h, x:x + target_w]
    
    elif resolution is not None:
        target_w, target_h = resolution  # Unpack width and height
        # image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (target_w, target_h))
        image = np.clip(image, 0, 1)


    image = depthanything_preprocess(image, color_aug=kwargs.get('color_aug', False))
    return image, _current_crop

def _load_and_process_depth(inverse_depth, image_shape, _current_crop, crop_type=None, resolution=None, resize_factor=1.0, **kwargs):
    # function takes inverse depth
    depth = inverse_depth

    valid_mask = (depth > 0).astype(np.uint8)

    if crop_type is not None and resolution is not None:
        target_w, target_h = resolution  # Unpack width and height
        h, w = depth.shape
        
        
        new_h, new_w = int(h * resize_factor), int(w * resize_factor)
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
        
        if crop_type == 'center':
            # First resize - use max scale to maintain aspect ratio while ensuring the image covers the target size
            # This matches the image processing logic
            scale = max(target_h / h, target_w / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Then center crop to target_w x target_h
            # Use the same crop coordinates as stored in _current_crop
            # if _current_crop is not None:
            #     start_y, start_x, crop_h, crop_w = _current_crop
            #     depth = depth[start_y:start_y + crop_h, start_x:start_x + crop_w]
            # else:
                # Fallback if _current_crop is not provided
            start_y = (new_h - target_h) // 2
            start_x = (new_w - target_w) // 2
            depth = depth[start_y:start_y + target_h, start_x:start_x + target_w]
        
        elif crop_type == 'random':
            assert _current_crop is not None, "Current crop must be provided for random crop"
            # new_h, new_w = int(h * resize_factor), int(w * resize_factor)
            # depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # h, w = new_h, new_w
            
            # Then ensure it's at least target_w x target_h
            scale = max(target_h / h, target_w / w)
            if scale > 1:
                new_h, new_w = int(h * scale), int(w * scale)
                depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w
            
            # Use same crop coordinates as image
            y, x = _current_crop['y'], _current_crop['x']
            depth = depth[y:y + target_h, x:x + target_w]


    # Resize to match image shape
    depth = cv2.resize(depth, (image_shape[2], image_shape[1]), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(depth).float()
    #valid_mask = cv2.resize(valid_mask, (image_shape[2], image_shape[1]), interpolation=cv2.INTER_NEAREST)
    # erode mask by 3 pixels
    # kernel = np.ones((3, 3), np.uint8)
    # valid_mask = cv2.erode(valid_mask, kernel, iterations=3)
    # # convert back to float for multiplication
    # valid_mask = valid_mask.astype(np.float32)

    # return torch.from_numpy(depth*valid_mask).float()



class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])
        
        # resize sample
        sample["image"] = cv2.resize(sample["image"], (width, height), interpolation=self.__image_interpolation_method)

        if self.__resize_target:
            if "depth" in sample:
                sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)
                
            if "mask" in sample:
                sample["mask"] = cv2.resize(sample["mask"].astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        
        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample
