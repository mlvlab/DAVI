import os
import functools
import numpy as np

import torch as th
import torchvision.utils as vtils
from PIL import Image as PILImage

from . import logger
from .resample import UniformSampler

from utils.noise import get_noise
from utils.tools import check_dims
from utils.svd_operators import get_degradation

from accelerate import Accelerator
from ema_pytorch import EMA
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from piq import LPIPS
from .losses_perceptual import LPIPSWithDiscriminator2D
import wandb

class TrainLoop:
    def __init__(
        self,
        *,
        teacher_model,
        student_model,
        implicit_model,
        diffusion,
        data,
        y_data,
        eval_data,
        micro_batch,
        global_batch,
        deg,
        lr,
        s_lr,
        weight_con,
        t_ikl,
        reg_coeff,
        perturb_h,
        save_dir,
        total_training_steps,
        total_training_iters,
        test_interval,
        resume_checkpoint,
        train_eval_interval=500,
        use_fp16=False,
        use_wandb=False,
        model_path=None,
        dataset_name='ffhq',
        wandb_project=None,
    ):  
        # Model and Diffusion
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.implicit_model = implicit_model
        self.total_training_steps = total_training_steps
        self.total_training_iters = total_training_iters
        self.diffusion = diffusion
        self.schedule_sampler = UniformSampler(self.diffusion)
        
        # Data
        self.data = data
        self.y_data = y_data
        self.eval_data = eval_data
        self.num_iters = len(self.data)
        self.eval_num_iters = len(self.eval_data)
        self.y_len = len(self.y_data)
        self.micro_batch = micro_batch
        self.global_batch = global_batch
        if self.micro_batch >= self.global_batch:
            self.grad_accum_steps = -1
        else:
            self.grad_accum_steps = self.global_batch / self.micro_batch
        logger.log(f'global batch: {self.global_batch}, accumulation_steps:{self.grad_accum_steps}')
        
        # Hyperparameters
        self.lr = lr
        self.s_lr = s_lr
        self.deg = deg
        self.weight_con = weight_con
        self.t_ikl = t_ikl
        self.reg_coeff = reg_coeff
        self.perturb_h = perturb_h

        self.test_interval = test_interval
        self.train_eval_interval = train_eval_interval
        self.save_dir = save_dir
            
        # Optimizer
        self.opt_d = th.optim.AdamW(self.implicit_model.parameters(), lr=self.lr)
        self.opt_s = th.optim.AdamW(self.student_model.parameters(), lr=self.s_lr)
        self.criterion = LPIPSWithDiscriminator2D(disc_in_channels=3, disc_weight=1.0) # patch-wise disc
        if dataset_name == 'ffhq':
            self.opt_disc = th.optim.AdamW(self.criterion.discriminator_2d.parameters(), lr=self.lr)
        else:
            self.opt_disc = th.optim.AdamW(self.criterion.discriminator_2d.parameters(), lr=self.lr * 10)

        self.step = 0
        self.resume_checkpoint = resume_checkpoint

        # Accelerator
        self.use_fp16 = use_fp16
        self.accelerator = Accelerator(
            split_batches = False,
            mixed_precision = 'fp16' if self.use_fp16 else 'no'
        )
        self.accelerator.native_amp = True if self.use_fp16 else False
        self.device = self.accelerator.device
        self.use_wandb = use_wandb
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.init(project=f'{wandb_project}_{dataset_name}', reinit=True)
            wandb.run.name = f'{self.deg}_wc{self.weight_con}_tikl{self.t_ikl}_gtcoeff{self.reg_coeff}_h{self.perturb_h}'
            
        if self.accelerator.is_main_process:
            self.ema = EMA(self.implicit_model, beta=0.9999, update_every=5)
        self.load(model_path)
        self.teacher_model, self.student_model, self.implicit_model = self.accelerator.prepare(self.teacher_model, self.student_model, self.implicit_model)
        self.data, self.opt_d, self.opt_s, self.criterion, self.opt_disc = self.accelerator.prepare(self.data, self.opt_d, self.opt_s, self.criterion, self.opt_disc)
        
        self.lpips = LPIPS(replace_pooling=False, reduction='none', mean=[0, 0, 0], std=[1, 1, 1])
        if self.deg == 'deno':
            self.noiser = get_noise(name='gaussian', sigma=0.20)
        else:
            self.noiser = get_noise(name='gaussian', sigma=0.05)
        self.operator = get_degradation(self.deg, self.accelerator.device)

    def get_measurements(self, clean):
        b, c, h, w = clean.shape
        
        if self.deg == 'sr_averagepooling' or self.deg == 'bicubic':
            h, w = int(h/4), int(w/4)
        elif self.deg == 'colorization':
            c = 1
            
        if self.deg == 'inpainting':
            blur = self.operator.A(clean).clone().detach()
            blur_pinv = self.operator.At(blur).view(b, c, h, w)
            blur = blur_pinv
        else:
            blur = self.operator.A(clean).view(b, c, h, w).clone().detach()
        blur = self.noiser(blur) # Additive noise
        blur = blur.to(self.device)
        
        if self.step % 100 == 0:
            vtils.save_image(blur, os.path.join(self.save_dir, 'train_measurement', f'step{self.step}_y.png'), normalize=True)
        
        return blur

    def run_loop(self):
        logger.log("Total {} epoch & each {} iter".format(self.total_training_steps, self.num_iters))
        indices = list(range(0,self.total_training_steps))
        for i in indices:
            if self.accelerator.is_main_process:
                logger.log("{} / {} epoch START".format(i, self.total_training_steps))

            for iters, (clean, _) in enumerate(self.data):
                
                if self.accelerator.is_main_process:
                    if self.step == self.resume_step:
                        logger.log(f"Training from STEP {self.step} START")
                    if self.step % 100 == 0:
                        logger.log(f'{(self.step)} iters START')
                if self.resume_checkpoint and self.step < self.resume_step:
                    self.step += 1
                    continue
                
                clean = clean * 2 - 1
                blur = self.get_measurements(clean)

                self.run_step(clean, blur, noise=None)

                # Check for Train images
                if (self.accelerator.is_main_process and self.step % self.train_eval_interval == 0): 
                    self.ema.ema_model.eval()
                    self.implicit_model.eval()

                    blur2 = check_dims(clean, blur, self.operator)
                    t = th.ones((blur2.shape[0],), dtype=th.int, device=clean.device) * 999
                    blur2 = self.diffusion.q_sample_i2sb(t, blur2, blur2, self.perturb_h)

                    with th.inference_mode():
                        img = self.diffusion.p_mean_variance(self.ema.ema_model, blur2, t, clip_denoised=False)['model_output']
                        img = self.diffusion.compute_pred_x0(t, blur2, img, False)
                        img = th.nn.functional.tanh(img)
                    
                    self.get_psnr_lpips(clean=clean, recon=img, train=True)
                    self.ema.ema_model.train()
                    self.implicit_model.train()
                self.accelerator.wait_for_everyone()

                if (self.step == self.total_training_iters or self.step % self.test_interval == 0): 
                    self.eval()
                self.accelerator.wait_for_everyone()


    def run_step(self, clean, blur, noise=None):
        # Update for student function phi
        self.forward_student(clean, blur, noise)
        self.opt_s.step()
        self.opt_s.zero_grad()
        self.opt_disc.step()
        self.opt_disc.zero_grad()
        
        # Update for student function 
        img, zeros = self.forward_implicit(clean, blur, noise)
        self.opt_d.step()
        self.opt_d.zero_grad()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.ema.update()
        self.step += 1
        return img, zeros

    def forward_student(self, clean, blur, noise):
        self.accelerator.wait_for_everyone()
        self.implicit_model.eval()
        self.student_model.train()
        self.opt_s.zero_grad()
        self.opt_disc.zero_grad()
        self.accelerator.wait_for_everyone()
        
        for i in range(0, blur.shape[0], self.micro_batch):
            micro_clean = clean[i:i+self.micro_batch]
            micro_blur = blur[i:i+self.micro_batch]
            t, weights = self.schedule_sampler.sample(micro_blur.shape[0], self.accelerator.device)
            
            with self.accelerator.autocast():
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.student_model,
                    self.implicit_model,
                    self.criterion,
                    micro_clean,
                    micro_blur,
                    t_ikl=self.t_ikl,
                    perturb_h=self.perturb_h,
                    model_kwargs=None,
                    operator=self.operator,
                    noise=noise,
                )
                losses = compute_losses()
                loss = (losses["loss"] * weights).mean()
                disc_loss = losses['gan_loss']

                if self.grad_accum_steps != -1:
                    loss = loss / self.grad_accum_steps
                    disc_loss = disc_loss / self.grad_accum_steps
                self.accelerator.backward(loss)
                self.accelerator.wait_for_everyone()
                self.accelerator.backward(disc_loss)
                self.accelerator.wait_for_everyone()

        if self.use_wandb and self.accelerator.is_main_process:
            wandb.log({'Score function loss': loss.item(), 'Disc loss': gan_loss.item()}, step=self.step)
    
    def forward_implicit(self, clean, blur, noise=None):
        self.accelerator.wait_for_everyone()
        self.implicit_model.train()
        self.student_model.eval()
        self.opt_d.zero_grad()
        self.accelerator.wait_for_everyone()
        
        for i in range(0, blur.shape[0], self.micro_batch):
            micro_clean = clean[i:i+self.micro_batch]
            micro_blur = blur[i:i+self.micro_batch]
            t, weights = self.schedule_sampler.sample(micro_blur.shape[0], self.accelerator.device)
            
            with self.accelerator.autocast():
                compute_losses = functools.partial(
                    self.diffusion.diffinstruct_losses,
                    self.teacher_model,
                    self.student_model,
                    self.implicit_model,
                    self.criterion,
                    micro_clean,
                    micro_blur,
                    t_ikl=self.t_ikl,
                    perturb_h=self.perturb_h,
                    model_kwargs=None,
                    operator=self.operator,
                    noise=noise,
                )
                losses, xt, micro = compute_losses()
                losses_1 = (losses['loss'] * micro).mean(dim=(1,2,3))

                kl_loss = (losses_1 * (weights)).mean()
                consistency_loss = losses['consistency']
                gt_consistency_loss = losses['gt_consistency']
                loss = kl_loss + self.weight_con * consistency_loss.mean() + self.reg_coeff * gt_consistency_loss
                gan_loss = losses['gan_loss']
                
                # if self.accelerator.num_processes == 1:
                #     import pdb; pdb.set_trace()
                #     disc_weight = self.calculate_adaptive_weight(gt_consistency_loss, gan_loss, self.implicit_model.output_blocks[-1][0].out_layers[-1].weight)
                # else: 
                disc_weight = self.calculate_adaptive_weight(gt_consistency_loss, gan_loss, self.implicit_model.module.output_blocks[-1][0].out_layers[-1].weight)
                disc_weight = th.clip(disc_weight, 0.01, 10)
                loss += disc_weight * gan_loss
                
                if self.grad_accum_steps != -1:
                    loss = loss / self.grad_accum_steps
                self.accelerator.backward(loss)
                self.accelerator.wait_for_everyone()  

        if self.use_wandb and self.accelerator.is_main_process:
            wandb.log({'KL loss': kl_loss.item(), 'Con loss': consistency_loss.mean().item(), 'Total loss': loss.item(), 'Gan loss': gan_loss.item(), 'disc weight': disc_weight.item()}, step=self.step)

        return micro, None

    def calculate_adaptive_weight(self, loss1, loss2, last_layer=None):
        loss1_grad = th.autograd.grad(loss1, last_layer, retain_graph=True)[0]
        loss2_grad = th.autograd.grad(loss2, last_layer, retain_graph=True)[0]
        d_weight = th.norm(loss1_grad) / (th.norm(loss2_grad) + 1e-4)
        d_weight = th.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def save(self):
        if not self.accelerator.is_main_process:
            return
        data = {
            'step' : self.step,
            'implicit_func' : self.accelerator.get_state_dict(self.implicit_model),
            'implicit_func_ema' : self.ema.state_dict(),
            'student_func' : self.accelerator.get_state_dict(self.student_model),
            'opt_d' : self.opt_d.state_dict(),
            'opt_s' : self.opt_s.state_dict(),
            'opt_disc' : self.opt_disc.state_dict(),
            'criterion_2d' : self.accelerator.get_state_dict(self.criterion),
            'scaler' : self.accelerator.scaler.state_dict(),
        }
        th.save(data, os.path.join(self.save_dir, 'model_ckpt', f'model-ema-{self.step}.pt'))
        
    def load(self, pretrained_ckpt):
        ckpt = th.load(pretrained_ckpt, map_location='cpu')
        self.teacher_model.load_state_dict(ckpt)
        self.teacher_model.eval()
        
        if self.resume_checkpoint:
            resume_ckpt = th.load(self.resume_checkpoint, map_location='cpu')
            logger.log(f"resume checkpoint from {self.resume_checkpoint}...")
            self.resume_step = resume_ckpt['step']
            self.implicit_model.load_state_dict(resume_ckpt['implicit_func'])
            if self.accelerator.is_main_process:
                self.ema.load_state_dict(resume_ckpt['implicit_func_ema'])
            self.student_model.load_state_dict(resume_ckpt['student_func'])
            self.opt_d.load_state_dict(resume_ckpt['opt_d'])
            self.opt_s.load_state_dict(resume_ckpt['opt_s'])
            self.opt_disc.load_state_dict(resume_ckpt['opt_disc'])
            self.criterion.load_state_dict(resume_ckpt['criterion_2d'])
            self.accelerator.scaler.load_state_dict(resume_ckpt['scaler'])
        else:
            self.resume_step = 0
            self.student_model.load_state_dict(ckpt)
            self.implicit_model.load_state_dict(ckpt)            
        if self.accelerator.is_main_process:
            self.ema.to(self.device)
            
    def eval(self):
        if not self.accelerator.is_main_process:
            return
        self.save()
        self.ema.ema_model.eval()
        self.implicit_model.eval()
        
        recon_dir = os.path.join(self.save_dir, 'recon', f'step{self.step}')
        os.makedirs(recon_dir, exist_ok=True)
        
        avg_psnr, avg_lpips = [],[]
        for idx, (clean, blur) in enumerate(zip(self.eval_data, self.y_data)):
            clean, _ = clean
            clean = clean * 2 - 1
            clean = clean.to(self.device)
            blur = blur[:,0]
            blur = blur.to(self.device)

            blur2 = check_dims(clean, blur, self.operator)
            t = th.ones((blur2.shape[0],), dtype=th.int, device=clean.device) * 999
            blur2 = self.diffusion.q_sample_i2sb(t, blur2, blur2, self.perturb_h)
            
            with self.accelerator.autocast():
                with th.no_grad():  
                    img = self.diffusion.p_mean_variance(self.ema.ema_model, blur2, t, clip_denoised=False)['model_output']
                    img = self.diffusion.compute_pred_x0(t, blur2, img, False)
                    img = th.nn.functional.tanh(img)
                    
            psnr_list, lpips_list = self.get_psnr_lpips(clean=clean, recon=img, train=False, start_idx=len(avg_psnr))
            avg_psnr.extend(psnr_list)
            avg_lpips.extend(lpips_list)
    
        self.ema.ema_model.train()
        self.implicit_model.train() 

        logger.log('steps %d: TEST AVG PSNR: %.3f / LPIPS: %.4f' % (self.step, np.mean(avg_psnr), np.mean(avg_lpips)))
        if self.use_wandb:
            wandb.log({'TEST PSNR':np.mean(avg_psnr), 'TEST LPIPS':np.mean(avg_lpips)}, step=self.step)

    def get_psnr_lpips(self, clean, recon, train=False, start_idx=0):
        assert clean.shape == recon.shape
        b, c, h, w = recon.shape
        
        if train:
            print_script = 'Train'
        else:
            print_script = 'Test'
            recon_dir = os.path.join(self.save_dir, 'recon', f'step{self.step}')
            os.makedirs(recon_dir, exist_ok=True)
        
        psnr_list, lpips_list = [],[]
        for i in range(b):
            psnr = psnr_loss((clean[i].cpu().numpy() + 1.)/2., (recon[i].detach().cpu().numpy() + 1.)/2.)  # psnr data range [0.1]
            lpips_loss = self.lpips(clean[i].reshape(1, c, h, w), recon[i].reshape(1, c, h, w)).item() # lpips range [-1.1] with mean [0,0,0] and std [1,1,1]
            psnr_list.append(psnr)
            lpips_list.append(lpips_loss)
            logger.log('steps %d: %s  %dth image PSNR: %.3f / LPIPS: %.4f' % (self.step, print_script, start_idx+i, psnr, lpips_loss))
            
            recon_ = (recon[i] + 1.)/2.
            image_np = recon_.data.cpu().numpy().transpose(1,2,0)
            image_np = PILImage.fromarray((image_np * 255).astype(np.uint8))

            if train:
                image_np.save(os.path.join(self.save_dir, 'train_recon', f'step{self.step}_{i}.png'))
                if self.use_wandb:
                    wandb.log({'Evaluation': wandb.Image(image_np)}, step=self.step)
                    wandb.log({'Train PSNR':psnr, 'Train lpips': lpips_loss}, step=self.step)
            else:
                image_np.save(os.path.join(recon_dir, f'{start_idx+i}.png'))
                vtils.save_image(clean[i], os.path.join(self.save_dir, 'label', f'{start_idx+i}.png'), normalize=True)
        
        return psnr_list, lpips_list