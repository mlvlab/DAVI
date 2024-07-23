import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NpyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.npy_file_paths = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        self.npy_file_paths = sorted(self.npy_file_paths)
        
    def __len__(self):
        return len(self.npy_file_paths)
    
    def __getitem__(self, idx):
        file_path = self.npy_file_paths[idx]
        data = np.load(file_path)
        tensor_data = torch.from_numpy(data)
        return tensor_data


def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w

def set_sq_bbox(img, mask_shape, image_size=256):
    """Generate a fixed sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    
    center_h, center_w = H //2, W // 2
    mask_half_h, mask_half_w = h // 2, w // 2
    
    start_h = center_h - mask_half_h
    start_w = center_w - mask_half_w
    
    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., start_h:start_h+h, start_w:start_w+w] = 0

    return mask

class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                                mask_shape=(mask_h, mask_w),
                                image_size=self.image_size,
                                margin=self.margin)
        return mask, t, tl, w, wh

    def _set_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask = set_sq_bbox(img,
                            mask_shape=(mask_h, mask_w),
                            image_size=self.image_size)
        return mask
    
    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask = self._set_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask

def unnormalize(img, s=0.95):
    scaling = torch.quantile(img.abs(), s)
    return img / scaling

def normalize(img, s=0.95):
    scaling = torch.quantile(img.abs(), s)
    return img * scaling

def dynamic_thresholding(img, s=0.95):
    img = normalize(img, s=s)
    return torch.clip(img, -1., 1.)