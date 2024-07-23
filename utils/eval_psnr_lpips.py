import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch as th
import matplotlib.pyplot as plt

import torchvision.datasets as dsets
from torch.utils import data
from torchvision import transforms
from piq import LPIPS
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from tools import CenterCropLongEdge, cycle

# Set device and data root
device = 'cuda:0'

label_root = Path('data/ffhq_1K')
recon_root = Path('results/')


lpips = LPIPS(replace_pooling=False, reduction='none', mean=[0, 0, 0], std=[1, 1, 1])
transform_list = transforms.Compose([
    CenterCropLongEdge(),
    transforms.Resize((256,256)),
    transforms.ToTensor()
])
l_dataset = dsets.ImageFolder(label_root, transform=transform_list)
l_data = data.DataLoader(l_dataset, batch_size=1, shuffle=False)
datasize = len(l_data)
l = cycle(l_data)

psnr_list, lpips_list = [], []

for idx in tqdm(range(datasize)):
    gt, _ = next(l)
    gt = gt.to(device)

    rc = plt.imread(os.path.join(recon_root, '{}.png'.format(idx)))[:,:,:3]
    rc = th.from_numpy(rc).permute(2, 0, 1).unsqueeze(0).to(device)
    rc = rc.to(device)

    psnr = psnr_loss(gt[0].cpu().numpy(), rc[0].cpu().numpy())
    psnr_list.append(psnr.item())
    
    with th.no_grad():
        lpips_score = lpips((gt*2-1), (rc*2-1))
    lpips_list.append(lpips_score.item())
    
print("AVG PSNR: %.3f / LPIPS: %.4f" % (np.mean(psnr_list), np.mean(lpips_list)))
