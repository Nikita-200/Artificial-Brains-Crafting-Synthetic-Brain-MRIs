import os
import torch
from ignite.engine import Engine
from ignite.metrics import FID
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import ConcatDataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchvision.utils as vutils
from torchvision.transforms import GaussianBlur
import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import json

# from dataset_loader import get_dataloaders
from diffusion_model import SliceDiffLite
from data import get_dataloaders_from_jpeg
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
from torchvision.models.inception import inception_v3
from torchvision.transforms import GaussianBlur
from skimage.metrics import peak_signal_noise_ratio as psnr

from torchvision.utils import save_image
import torch.nn.functional as F
import os, torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
###### print and log FID, SSIM, PSNR, Test loss, use archive root in testing
# === Diffusion hyperparams ===
def diffusion_loss(pred_noise, true_noise, seg_mask):
    # Weighted MSE loss
    mse_loss = F.mse_loss(pred_noise, true_noise)
    
    # Edge-aware loss using segmentation mask
    edges = F.max_pool2d(seg_mask, 3, stride=1, padding=1) - seg_mask
    edge_loss = F.l1_loss(pred_noise*edges, true_noise*edges)
    # return 1.0*mse_loss + 0.0*edge_loss
    
    return 0.85*mse_loss + 0.15*edge_loss
    
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
    
def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

T = 1000
betas = linear_beta_schedule(T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# === Resize and repeat grayscale for InceptionV3 ===
def interpolate(batch):
    out = []
    for img in batch:
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        pil = to_pil_image(img)
        out.append(ToTensor()(pil.resize((299, 299), Image.BILINEAR)))
    return torch.stack(out)

blur = GaussianBlur(kernel_size=5, sigma=(1.5, 1.5))

# === Reverse DDPM Sampling ===
# def sample(model, seg_mask, real_img, T, alphas, betas, alphas_cumprod, device):
def sample(model, seg_mask, real_img):
    model.eval()
    blurred_img = blur((real_img + 1) / 2.0) 
    blurred_img = blurred_img * 2 - 1 
    with torch.no_grad():
        x_t = torch.randn_like(real_img).to(device)
        for t_inv in reversed(range(0, T)):
            t = torch.full((seg_mask.size(0),), t_inv, device=device).long()
            pred_noise = model(x_t, seg_mask.float(), blurred_img.float(), t)
            alpha = alphas[t][:, None, None, None]
            beta = betas[t][:, None, None, None]
            x_t = (1 / torch.sqrt(alpha)) * (x_t - (beta / torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]) * pred_noise)

            # Fixed version should include:
            if t_inv > 0:
                sigma = torch.sqrt(betas[t])[:, None, None, None]  # Critical fix
                # sigma = torch.sqrt(betas[t])
                x_t += sigma * torch.randn_like(x_t)

        x_t = (x_t + 1) / 2.0  # scale [-1,1] → [0,1]
            # if t_inv % 50 == 0:
            #     vutils.save_image(x_t[:8], f"denoise_step_{t_inv}.png", normalize=True)
        

    return x_t

# === Load model ===
model = SliceDiffLite(in_channels=1, cond_channels=1, time_emb_dim=64).to(device)
model.load_state_dict(torch.load("checkpoints_ddpm_fid/best_generator_t1C_all500.pth"))
# model ckpt on FID is BAAAAD
model.eval()

# === Load combined val + test set ===
test_loader = get_dataloaders_from_jpeg(batch_size=8) ######################## CHANGE TO TEST ONLY 
# combined_set = ConcatDataset([val_loader.dataset, test_loader.dataset])
# combined_loader = DataLoader(combined_set, batch_size=8, shuffle=False, drop_last=True)
combined_loader = test_loader

# === FID Evaluator ===
fid_metric = FID(device=device)

model.eval()
num_saved = 0

inception = inception_v3(pretrained=True, transform_input=False).to(device)
inception.eval()

def get_feats(imgs):
    imgs = interpolate(imgs)
    with torch.no_grad():
        return inception(imgs.to(device)).detach().cpu().numpy()

# === Evaluation ===
total_loss = 0.0
total_psnr = 0.0
total_ssim = 0.0
real_feats = []
fake_feats = []
num_samples = 0
global_index = 0

import matplotlib.pyplot as plt

vis_save_dir = "testing_T1C"
os.makedirs(vis_save_dir, exist_ok=True)

def save_subplot(seg, real, fake, idx):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(seg.squeeze(), cmap='gray')
    axs[0].set_title("Seg Mask")
    axs[1].imshow(real.squeeze(), cmap='gray')
    axs[1].set_title("Real")
    axs[2].imshow(fake.squeeze(), cmap='gray')
    axs[2].set_title("Fake")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_save_dir, f"triplet_{idx:03d}.png"))
    plt.close()



for real_img, seg_mask in tqdm(test_loader, desc="Evaluating"):
    real_img = real_img.to(device)
    seg_mask = seg_mask.to(device)

    real_img = real_img * 2 - 1
    seg_mask = seg_mask * 2 - 1

    # === Generate fake ===
    fakes = sample(model, seg_mask, real_img)

    # === Compute denoising loss ===
    t_rand = torch.randint(0, T, (real_img.size(0),), device=device).long()
    noise = torch.randn_like(real_img)
    x_t = torch.sqrt(alphas_cumprod[t_rand])[:, None, None, None] * real_img + torch.sqrt(1 - alphas_cumprod[t_rand])[:, None, None, None] * noise
    blurred_img = blur((real_img + 1) / 2.0) 
    blurred_img = blurred_img * 2 - 1 
    pred_noise = model(x_t, seg_mask, blurred_img, t_rand)
    loss = diffusion_loss(pred_noise, noise, seg_mask)
    # loss = F.mse_loss(pred_noise, noise)
    total_loss += loss.item() * real_img.size(0)

    # === Compute PSNR and SSIM ===
    real_np = ((real_img + 1) / 2.0).cpu().numpy()
    fake_np = fakes.cpu().numpy()
    for i in range(real_np.shape[0]):
        # Save triplet subplot
        save_subplot(((seg_mask[i] + 1) / 2.0).cpu().numpy(),((real_img[i] + 1) / 2.0).cpu().numpy(),fakes[i].cpu().numpy(),global_index)
        global_index += 1

        total_psnr += compute_psnr(real_np[i, 0], fake_np[i, 0], data_range=1.0)
        total_ssim += compute_ssim(real_np[i, 0], fake_np[i, 0], data_range=1.0)

    # === Extract Inception features ===
    real_feats.append(get_feats((real_img + 1) / 2.0))
    fake_feats.append(get_feats(fakes))
    num_samples += real_img.size(0)

# Average metrics
avg_loss = total_loss / num_samples
print("Avg loss", avg_loss)
avg_psnr = total_psnr / num_samples
print("Avg PSNR", avg_psnr)
avg_ssim = total_ssim / num_samples
print("Avg SSIM", avg_ssim)

results_dict = {
    "FID": float(fid),
    "Avg_PSNR": float(avg_psnr),
    "Avg_SSIM": float(avg_ssim),
    "Test_Loss": float(avg_loss),
    "Num_Samples": int(num_samples),
    "Beta_Schedule": "linear",
    "T": T
}

with open("fid_t1c.json", "w") as f:
    json.dump(results_dict, f, indent=4)

# === FID Calculation ===
real_feats = np.concatenate(real_feats, axis=0)
fake_feats = np.concatenate(fake_feats, axis=0)
mu1, sigma1 = real_feats.mean(0), np.cov(real_feats, rowvar=False)
mu2, sigma2 = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
if np.iscomplexobj(covmean):
    covmean = covmean.real
fid = ((mu1 - mu2) ** 2).sum() + np.trace(sigma1 + sigma2 - 2 * covmean)

print("Final FID score", fid)