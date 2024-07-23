# [FFHQ] Gaussian deblur
accelerate launch --multi_gpu --num_processes=4 train.py --data_dir data/ffhq_49K/ --model_path model/ffhq_10m.pt  --test_interval 1000 --save_dir results --global_batch 8 --micro_batch 8 --deg gaussian --t_ikl 400 --weight_con 0.5 --reg_coeff 0.25 --perturb_h 0.1

# [FFHQ] 4x Super-resolution
accelerate launch --multi_gpu --num_processes=4 train.py --data_dir data/ffhq_49K/ --model_path model/ffhq_10m.pt  --test_interval 1000 --save_dir results --global_batch 8 --micro_batch 8 --deg sr_averagepooling --t_ikl 1000 --weight_con 0.1 --reg_coeff 1.0 --perturb_h 0.1

# [FFHQ] Box Inpainting
accelerate launch --multi_gpu --num_processes=4 train.py --data_dir data/ffhq_49K/ --model_path model/ffhq_10m.pt  --test_interval 1000 --save_dir results --global_batch 8 --micro_batch 8 --deg inpainting --t_ikl 1000 --weight_con 0.5 --reg_coeff 1.0 --perturb_h 0.1

# [FFHQ] Denoising
accelerate launch --multi_gpu --num_processes=4 train.py --data_dir data/ffhq_49K/ --model_path model/ffhq_10m.pt  --test_interval 1000 --save_dir results --global_batch 8 --micro_batch 8 --deg deno --t_ikl 1000 --weight_con 0.2 --reg_coeff 0.1 --perturb_h 0.1

# [FFHQ] Colorization
accelerate launch --multi_gpu --num_processes=4 train.py --data_dir data/ffhq_49K/ --model_path model/ffhq_10m.pt  --test_interval 1000 --save_dir results --global_batch 8 --micro_batch 8 --deg colorization --t_ikl 1000 --weight_con 1 --reg_coeff 0.25 --perturb_h 0.1



# [ImageNet] Gaussian deblur
accelerate launch --multi_gpu --num_processes=4 train.py --data_dir data/imagenet_130K/ --model_path model/256x256_diffusion_uncond.pt --test_interval 2000 --save_dir results --global_batch 12 --micro_batch 3 --deg gaussian --t_ikl 400 --weight_con 0.5 --reg_coeff 0.5 --perturb_h 0.1

# [ImageNet] 4x Super-resolution
accelerate launch --multi_gpu --num_processes=4 train.py --data_dir data/imagenet_130K/ --model_path model/256x256_diffusion_uncond.pt --test_interval 2000 --save_dir results --global_batch 12 --micro_batch 3 --deg sr_averagepooling --t_ikl 1000 --weight_con 0.075 --reg_coeff 0.25 --perturb_h 0.01

# [ImageNet] Box Inpainting
accelerate launch --multi_gpu --num_processes=4 train.py --data_dir data/imagenet_130K/ --model_path model/256x256_diffusion_uncond.pt --test_interval 2000 --save_dir results --global_batch 12 --micro_batch 3 --deg inpainting --t_ikl 400 --weight_con 0.01 --reg_coeff 0.1 --perturb_h 0.01

# [ImageNet] Denoising
accelerate launch --multi_gpu --num_processes=4 train.py --data_dir data/imagenet_130K/ --model_path model/256x256_diffusion_uncond.pt --test_interval 2000 --save_dir results --global_batch 12 --micro_batch 3 --deg deno --t_ikl 600 --weight_con 0.1 --reg_coeff 1.0 --perturb_h 0.01

#  [ImageNet] Colorization
accelerate launch --multi_gpu --num_processes=4 train.py --data_dir data/imagenet_130K/ --model_path model/256x256_diffusion_uncond.pt --test_interval 2000 --save_dir results --global_batch 12 --micro_batch 3 --deg colorization --t_ikl 1000 --weight_con 0.5 --reg_coeff 0.1 --perturb_h 0.1