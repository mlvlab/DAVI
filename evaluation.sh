# [FFHQ] Gaussian deblur
accelerate launch --num_processes=1 eval.py --eval_dir data/ffhq_1K --save_dir results --deg gaussian --perturb_h 0.1 --ckpt model/official_ckpt/ffhq/gaussian_ema.pt

# [FFHQ] 4x Super-resolution
accelerate launch --num_processes=1 eval.py --eval_dir data/ffhq_1K --save_dir results --deg sr_averagepooling --perturb_h 0.1 --ckpt model/official_ckpt/ffhq/sr_averagepooling_ema.pt

# [FFHQ] Box Inpainting
accelerate launch --num_processes=1 eval.py --eval_dir data/ffhq_1K --save_dir results --deg inpainting --perturb_h 0.1 --ckpt model/official_ckpt/ffhq/inpainting_ema.pt

# [FFHQ] Denoising
accelerate launch --num_processes=1 eval.py --eval_dir data/ffhq_1K --save_dir results --deg deno --perturb_h 0.1 --ckpt model/official_ckpt/ffhq/deno_ema.pt

# [FFHQ] Colorization
accelerate launch --num_processes=1 eval.py --eval_dir data/ffhq_1K --save_dir results --deg colorization --perturb_h 0.1 --ckpt model/official_ckpt/ffhq/colorization_ema.pt



# [ImageNet] Gaussian deblur
accelerate launch --num_processes=1 eval.py --eval_dir data/imagenet_val_1K --save_dir results --deg gaussian --perturb_h 0.1 --ckpt model/official_ckpt/imagenet/gaussian_ema.pt

# [ImageNet] 4x Super-resolution
accelerate launch --num_processes=1 eval.py --eval_dir data/imagenet_val_1K --save_dir results --deg sr_averagepooling --perturb_h 0.01 --ckpt model/official_ckpt/imagenet/sr_averagepooling_ema.pt

# [ImageNet] Box Inpainting
accelerate launch --num_processes=1 eval.py --eval_dir data/imagenet_val_1K --save_dir results --deg inpainting --perturb_h 0.01 --ckpt model/official_ckpt/imagenet/inpainting_ema.pt

# [ImageNet] Denoising
accelerate launch --num_processes=1 eval.py --eval_dir data/imagenet_val_1K --save_dir results --deg deno --perturb_h 0.01 --ckpt model/official_ckpt/imagenet/deno_ema.pt

# [ImageNet] Colorization
accelerate launch --num_processes=1 eval.py --eval_dir data/imagenet_val_1K --save_dir results --deg colorization --perturb_h 0.01 --ckpt model/official_ckpt/imagenet/colorization_ema.pt
