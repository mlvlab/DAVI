import os
import numpy as np
import argparse
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vtils

from tools import random_seed, CenterCropLongEdge
from noise import get_noise
from svd_operators import get_degradation

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--deg', type=str, default='gaussian')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str, default='data/y_npy')
    return parser

def get_measurements(clean, deg, operator, noiser):
    b, c, h, w = clean.shape
    
    if deg == 'sr_averagepooling' or deg == 'bicubic':
        h, w = int(h/4), int(w/4)
    elif deg == 'colorization':
        c = 1
        
    if deg == 'inpainting':
        blur = operator.A(clean).clone().detach()
        blur_pinv = operator.At(blur).view(b, c, h, w)
        blur = blur_pinv
    else:
        blur = operator.A(clean).view(b, c, h, w).clone().detach()
    blur = noiser(blur) # Additive noise

    return blur


def main():
    args = create_argparser().parse_args()
    
    random_seed(args.seed)
    
    dataset_name = 'ffhq' if 'ffhq' in args.data_dir else 'imagenet'
    deg = args.deg
    args.save_dir = os.path.join(args.save_dir, f'{dataset_name}_1k_npy', f'{deg}')
    os.makedirs(args.save_dir)
    os.makedirs(os.path.join(args.save_dir, 'label', 'data'))
    os.makedirs(os.path.join(args.save_dir, 'measurement'))
    os.makedirs(os.path.join(args.save_dir, 'measurement_img'))
    
    # Dataloader
    transform_list = transforms.Compose([
        CenterCropLongEdge(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = dsets.ImageFolder(args.data_dir, transform=transform_list)
    data = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    
    if deg == 'deno':
        noiser = get_noise(name='gaussian', sigma=0.20)
    else:
        noiser = get_noise(name='gaussian', sigma=0.05)
    operator = get_degradation(deg, device='cpu')


    for iters, (clean, _) in enumerate(data):
        clean = clean * 2 - 1
        vtils.save_image(clean, os.path.join(args.save_dir, 'label', 'data', '{:04d}.png'.format(iters)), normalize=True)
        
        blur = get_measurements(clean, deg, operator, noiser)
        blur_np = blur.cpu().numpy()
        np.save(os.path.join(args.save_dir, 'measurement', '{:04d}'.format(iters)), blur_np)
        vtils.save_image(blur, os.path.join(args.save_dir, 'measurement_img', '{:04d}.png'.format(iters)), normalize=True)
        
    print(f"{dataset_name} measurement saved {args.save_dir}")
    
if __name__ == "__main__":
    main()