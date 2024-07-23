import os
import numpy as np
import torch as th
import torchvision.utils as vtils
from PIL import Image as PILImage

from utils.noise import get_noise
from utils.tools import check_dims, cycle
from utils.svd_operators import get_degradation

from accelerate import Accelerator
from ema_pytorch import EMA

class EvalLoop:
    def __init__(
        self,
        *,
        implicit_model,
        diffusion,
        batch_size,
        eval_data,
        y_data,
        perturb_h,
        ckpt,
        save_dir,
        deg,
    ):  
        self.implicit_model = implicit_model
        self.diffusion = diffusion
        
        self.batch_size = batch_size
        self.eval_data = eval_data
        if y_data is not None:
            self.y_data = cycle(y_data)
        else:
            self.y_data = y_data
        
        self.perturb_h = perturb_h
        self.save_dir = save_dir
        self.deg = deg

        self.accelerator = Accelerator(
            split_batches = False,
            mixed_precision = 'no'
        )
        self.accelerator.native_amp = False
        self.device = self.accelerator.device
        self.ema = EMA(self.implicit_model)
        self.load(ckpt)
        
        if self.deg == 'deno':
            self.noiser = get_noise(name='gaussian', sigma=0.20)
        else:
            self.noiser = get_noise(name='gaussian', sigma=0.05)
        self.operator = get_degradation(self.deg, self.device)

    def run_loop(self):

        img_cnt = 0
        for idx, (clean, _) in enumerate(self.eval_data):
            clean = clean * 2 - 1
            
            if self.y_data is None:
                blur = self.get_measurements(clean.to(self.device))
            else:
                blur = next(self.y_data)
                blur = blur[:,0].to(self.device)
            
            blur2 = check_dims(clean, blur, self.operator)
            t = th.ones((blur2.shape[0],), dtype=th.int, device=self.device) * 999
            blur2 = self.diffusion.q_sample_i2sb(t, blur2, blur2, self.perturb_h)

            with self.accelerator.autocast():
                with th.no_grad():  
                    img = self.diffusion.p_mean_variance(self.ema.ema_model, blur2, t, clip_denoised=False)['model_output']
                    img = self.diffusion.compute_pred_x0(t, blur2, img, False)
                    img = th.nn.functional.tanh(img)
            
            for i in range(clean.shape[0]):
                vtils.save_image(blur[i], os.path.join(self.save_dir, 'measurement', f'{img_cnt}.png'), normalize=True)
                recon = (img[i] + 1.)/2.
                image_np = recon.data.cpu().numpy().transpose(1,2,0)
                image_np = PILImage.fromarray((image_np * 255).astype(np.uint8))
                image_np.save(os.path.join(self.save_dir, 'recon', f'{img_cnt}.png'))
                img_cnt += 1
                
            print(f"{img_cnt} th image generated...")
            
    def load(self, ckpt):
        checkpoint = th.load(ckpt, map_location='cpu')
        self.ema.load_state_dict(checkpoint['implicit_func_ema'])
        self.ema.to(self.device)
        self.ema.ema_model.eval()

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

        return blur