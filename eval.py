import os
import argparse
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from utils.img_utils import NpyDataset

from cm import logger
from cm.script_util import (
    ffhq_model_and_ppb_diffusion_defaults,
    imagenet_model_and_ppb_diffusion_defaults,
    create_model_and_guided_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from cm.eval_util import EvalLoop
from utils.tools import random_seed, CenterCropLongEdge

def main():
    args = create_argparser().parse_args()

    random_seed(args.seed)
    
    deg_type = args.deg
    dataset_name = 'ffhq' if 'ffhq' in args.eval_dir else 'imagenet'
    args.save_dir = os.path.join(args.save_dir, f'Evaluations_{deg_type}_{dataset_name}', f'seed{args.seed}')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'measurement'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'recon'), exist_ok=True)
    
    logger.configure(dir=args.save_dir)
    
    logger.log("creating model and diffusion...")
    if dataset_name == 'ffhq':
        model_and_diffusion_defaults = ffhq_model_and_ppb_diffusion_defaults
    elif dataset_name == 'imagenet':
        model_and_diffusion_defaults = imagenet_model_and_ppb_diffusion_defaults
    
    model_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
    model_dict.update(model_and_diffusion_defaults())
    implicit_model, diffusion = create_model_and_guided_diffusion(**model_dict) # implicit function

    logger.log("creating data loader...")
    transform_list = transforms.Compose([
        CenterCropLongEdge(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    
    eval_dataset = dsets.ImageFolder(args.eval_dir, transform=transform_list)
    eval_data = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)
    if args.y_dir is not None:
        y_dir = os.path.join(args.y_dir, f'{dataset_name}_1k_npy/{args.deg}')
        y_dataset = NpyDataset(data_dir=os.path.join(y_dir, 'measurement'))
        y_data = torch.utils.data.DataLoader(y_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)
    else:
        y_data = None

    logger.log("Evaluation...")
    EvalLoop(
        implicit_model=implicit_model,
        diffusion=diffusion,
        batch_size=args.batch_size,
        eval_data=eval_data,
        y_data=y_data,
        perturb_h=args.perturb_h,
        ckpt=args.ckpt,
        save_dir=args.save_dir,
        deg=args.deg,
    ).run_loop()


def create_argparser():
    defaults = dict(
        eval_dir="",
        save_dir="results/",
        batch_size=10,
        seed=777,
    )
    defaults.update(ffhq_model_and_ppb_diffusion_defaults())
    defaults.update(imagenet_model_and_ppb_diffusion_defaults())
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--deg', type=str)
    parser.add_argument('--perturb_h', type=float, default=0.1)
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--y_dir', type=str, default=None)

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

