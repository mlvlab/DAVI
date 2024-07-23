import os
import argparse
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from utils.img_utils import NpyDataset
import datetime

from cm import logger
from cm.script_util import (
    ffhq_model_and_ppb_diffusion_defaults,
    imagenet_model_and_ppb_diffusion_defaults,
    create_model_and_guided_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from cm.train_util import TrainLoop
from utils.tools import random_seed, CenterCropLongEdge, save_mkdirs

def main():
    args = create_argparser().parse_args()
    
    random_seed(args.seed)

    assert args.global_batch % args.micro_batch == 0
    logger.log(f"global_batch: {args.global_batch} / micro batch size {args.micro_batch}")

    dataset_name = 'ffhq' if 'ffhq' in args.data_dir else 'imagenet'
    args.save_dir = os.path.join(
        args.save_dir, 
        f'{dataset_name}_{args.deg}/gbs{args.global_batch}_bs{args.micro_batch}_lr{args.lr}_slr{args.s_lr}/wc{args.weight_con}-tikl{args.t_ikl}-reg_coeff{args.reg_coeff}-h{args.perturb_h}'
    )
    save_mkdirs(args.save_dir)
    logger.configure(dir=args.save_dir)
    logger.log("Current time: {}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")))
    
    logger.log("creating model and diffusion...")
    if dataset_name == 'ffhq':
        model_and_diffusion_defaults = ffhq_model_and_ppb_diffusion_defaults
    elif dataset_name == 'imagenet':
        model_and_diffusion_defaults = imagenet_model_and_ppb_diffusion_defaults
    
    model_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
    model_dict.update(model_and_diffusion_defaults())

    model, _ = create_model_and_guided_diffusion(**model_dict) # teacher model
    s_model, _ = create_model_and_guided_diffusion(**model_dict) # score function
    implicit_model, diffusion = create_model_and_guided_diffusion(**model_dict) # implicit function

    logger.log("creating data loader...")
    transform_list = transforms.Compose([
        CenterCropLongEdge(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    dataset = dsets.ImageFolder(args.data_dir, transform=transform_list)
    data = torch.utils.data.DataLoader(dataset, batch_size=args.global_batch, shuffle=True, pin_memory=True, drop_last=False)
    
    y_dir = os.path.join(args.y_dir, f'{dataset_name}_1k_npy/{args.deg}')
    y_dataset = NpyDataset(data_dir=os.path.join(y_dir, 'measurement'))
    y_data = torch.utils.data.DataLoader(y_dataset, batch_size=args.eval_batch, shuffle=False, pin_memory=True, drop_last=False)

    eval_dataset = dsets.ImageFolder(os.path.join(y_dir, 'label'), transform=transform_list)
    eval_data = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch, shuffle=False, pin_memory=True, drop_last=False)
    
    logger.log("Training...")
    TrainLoop(
        teacher_model=model,
        student_model=s_model,
        implicit_model=implicit_model,
        diffusion=diffusion,
        data=data,
        y_data=y_data,
        eval_data=eval_data,
        micro_batch=args.micro_batch,
        global_batch=args.global_batch,
        deg=args.deg,
        lr=args.lr,
        s_lr=args.s_lr,
        weight_con=args.weight_con,
        t_ikl=args.t_ikl,
        reg_coeff=args.reg_coeff,
        perturb_h=args.perturb_h,
        save_dir=args.save_dir,
        total_training_steps=args.total_training_steps,
        total_training_iters=args.total_training_iters,
        test_interval=args.test_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        use_wandb=args.use_wandb,
        model_path=args.model_path,
        dataset_name=dataset_name,
        wandb_project=args.wandb_project,
    ).run_loop()


def create_argparser():
    defaults = dict(
        y_dir="data/y_npy",
        save_dir="results/",
        data_dir="data/ffhq_49K",
        model_path="model/ffhq_10m.pt",
        test_interval=1000,
        total_training_steps=100,
        total_training_iters=2000000,
        global_batch=8,
        micro_batch=8,
        eval_batch=10,
        lr=1e-4,
        s_lr=1e-4,
        resume_checkpoint=None,
        seed=777,
    )
    defaults.update(ffhq_model_and_ppb_diffusion_defaults())
    defaults.update(imagenet_model_and_ppb_diffusion_defaults())
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--deg', type=str, default='gaussian')
    parser.add_argument('--use_wandb',  action='store_true') 
    parser.add_argument('--wandb_project', type=str, default='DAVI')

    parser.add_argument('--weight_con', type=float, default=1)
    parser.add_argument('--t_ikl', type=int, default=1000) 
    parser.add_argument('--perturb_h', type=float, default=0.1)
    parser.add_argument('--reg_coeff', type=float, default=0.1)
    
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()