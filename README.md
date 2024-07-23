# DAVI: Diffusion Prior-Based Amortized Variational Inference for Noisy Inverse Problems

<!-- ** [Paper] ()** -->

Sojin Lee*, Dogyun Park*, Inho Kong, Hyunwoo J. Kim†.

This repository is an official pytorch implementation of the **DAVI**: **D**iffusion Prior-Based **A**mortized **V**ariational **I**nference for Noisy Inverse Problems accepted at **ECCV 2024**.

Our framework enables efficient posterior sampling by **_a single evaluation of a neural network_** and **_generalization_** for both seen and unseen measurements without any optimization at test time.

<div align="center">
  <img src="asset/main.png" width="700px" />
</div>

## Setting

### 1. Clone the Repository

```
git clone https://github.com/mlvlab/DAVI.git
cd DAVI
```

### 2. Install Environment

```
conda create -n DAVI python==3.8
conda activate DAVI
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia

```

```
pip install accelerate ema_pytorch matplotlib piq scikit-image pytorch-fid wandb
```

### 3. Download Pre-trained models and Official Checkpoints

We utilize pre-trained models of [FFHQ (ffhq_10m.pt)](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh)(from [DPS](https://github.com/DPS2022/diffusion-posterior-sampling?tab=readme-ov-file)) and [ImageNet (256x256_diffusion_uncond.pt)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)(from [guided_diffusion](https://github.com/openai/guided-diffusion)).

From this [Google Drive Link](https://drive.google.com/drive/folders/1h8vKYwTYSshljBuW9NdBRJQSp_HBPZdA), download the checkpoints of FFHQ and ImageNet.

Put it pre-trained models into `model/` and our checkpoints into `model/official_ckpt/ffhq` or `model/official_ckpt/imagenet`.

### 4. Prepare Data

For amortized optimization, we utilize the FFHQ 49K dataset and the ImageNet 130K dataset, which are subsets of the training dataset used to train the pre-trained models. These are distinct sets from the validation datasets (ffhq_1K and imagenet_val_1K) used for evaluation.

- [FFHQ 256x256] `data/ffhq_1K`, `data/ffhq_49K`

  We download FFHQ dataset and resize it into 256x256 as following [ffhq-dataset
  public site](https://github.com/NVlabs/ffhq-dataset). We use 00000-00999 as the validation set (1K) and 01000-49999 (49K) as the training set.

- [ImageNet 256x256] `data/imagenet_val_1K`, `data/imagenet_130K`

  We download [ImageNet 100](https://www.kaggle.com/datasets/ambityga/imagenet100) and use its training set.

- [Measurements as numpy format]
  During amortized training, we load a subset of the training set to monitor the convergence of the training process.
  Prepare measurements from training set into `data/y_npy`.
  ```
  python utils/get_measurements.py --deg gaussian --data_dir data/ffhq_49K
  ```

## Overall directory

```
├── results
│
├── models
│ ├── ffhq_10m.pt # FFHQ for training
│ ├── 256x256_diffusion_uncond.pt # ImageNet for training
│ └── official_ckpt # For Evaluation
│     ├── ffhq
│     │   ├── gaussian_ema.pt
│     │   ├── sr_averagepooling_ema.pt
│     │   ├── ...
│     │   ├── ...
│     ├── imagenet
│     │   ├── gaussian_ema.pt
│     │   ├── sr_averagepooling_ema.pt
│     │   ├── ...
│     └── └── ...
│
├── data # including training set and evaluation set
│ ├── ffhq_1K # FFHQ evluation
│ ├── imagenet_val_1K # ImageNet evluation
│ ├── ffhq_49K # FFHQ training
│ ├── imagenet_130K # ImageNet training
│ └── y_npy
│         ├── ffhq_1k_npy
│         │   ├── gaussian
│         │   ├── sr_averagepooling
│         │   ├── ...
│         │   └── ...
│         ├── imagenet_val_1k_npy
│         │   ├── gaussian
│         │   ├── sr_averagepooling
│         │   ├── ...
└─────────└── └── ...
```

## Evaluation

### 1. Restore degraded images

```
accelerate launch --num_processes=1 eval.py --eval_dir data/ffhq_1K --deg gaussian --perturb_h 0.1 --ckpt model/official_ckpt/ffhq/gaussian_ema.pt
```

- You can specify directory of measurements `--y_dir data/y_npy`

### 2. Evaluate PSNR,LPIPS and FID

- PSNR and LPIPS
  ```
  python utils/eval_psnr_lpips.py
  ```
- FID: [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
  ```
  python -m pytorch_fid source_dir recon_dir
  ```

## Train with MultiGPU

- You can use `--use_wandb` to check training logs.

```
accelerate launch --multi_gpu --num_processes=4 train.py --data_dir data/ffhq_49K/ --model_path model/ffhq_10m.pt --deg gaussian --t_ikl 400 --weight_con 0.5 --reg_coeff 0.25 --perturb_h 0.1
```