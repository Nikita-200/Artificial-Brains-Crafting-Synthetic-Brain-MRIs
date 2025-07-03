import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm
from ignite.engine import Engine, Events
from ignite.metrics import FID
from ignite.handlers import ModelCheckpoint
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import GaussianBlur
import json
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# from dataset_loader_new import get_dataloaders
from diffusion_model import SliceDiffLite
from data import get_dataloaders_from_jpeg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(img1, img2):
    """
    Compute average PSNR and SSIM for batches of grayscale images.
    Assumes input images are in [0, 1] range and shape (B, 1, H, W).
    """
    psnr_vals, ssim_vals = [], []
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    for i in range(img1_np.shape[0]):
        p = psnr(img1_np[i, 0], img2_np[i, 0], data_range=1.0)
        s = ssim(img1_np[i, 0], img2_np[i, 0], data_range=1.0)
        psnr_vals.append(p)
        ssim_vals.append(s)
    return sum(psnr_vals) / len(psnr_vals), sum(ssim_vals) / len(ssim_vals)


# === DDPM Noise Scheduler ===
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

# Hybrid Loss Function (Add to training script)
def diffusion_loss(pred_noise, true_noise, seg_mask):
    # Weighted MSE loss
    mse_loss = F.mse_loss(pred_noise, true_noise)
    
    # Edge-aware loss using segmentation mask
    edges = F.max_pool2d(seg_mask, 3, stride=1, padding=1) - seg_mask
    edge_loss = F.l1_loss(pred_noise*edges, true_noise*edges)
    # return 1.0*mse_loss + 0.0*edge_loss
    
    return 0.85*mse_loss + 0.15*edge_loss

T = 1000 #100 before ---- try 64 base channels
betas = linear_beta_schedule(T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

# === Forward diffusion sampling ===
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alpha = sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# === FID Utility: Resize to 299x299 and repeat grayscale to RGB ===
def interpolate(batch):
    out = []
    for img in batch:
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        pil_img = TF.to_pil_image(img)
        pil_resized = pil_img.resize((299, 299), Image.BILINEAR)
        out.append(ToTensor()(pil_resized))
    return torch.stack(out)

# === Load data ===
train_loader, val_loader = get_dataloaders_from_jpeg(batch_size=8)

# === Model + optimizer ===
model = SliceDiffLite(in_channels=1, cond_channels=1, time_emb_dim=64).to(device) #time_emb_dim was 128 before
model.apply(init_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6) #lr = 1e-3 before
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

# === Training ===
num_epochs = 500
save_dir = "checkpoints_ddpm_fid"
os.makedirs(save_dir, exist_ok=True)
best_fid = float("inf")
best_loss = float("inf")
best_val_loss = float("inf")

early_stopping_patience = 10  # Stop if no improvement for 10 epochs
epochs_without_improvement = 0

blur = GaussianBlur(kernel_size=5, sigma=(1.5, 1.5))

metrics_log = []
# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    total_loss = 0

    for batch_idx, (real_img, seg_mask) in enumerate(pbar):
        # print(f"\n[Batch {batch_idx}]")
        seg_mask, real_img = seg_mask.to(device), real_img.to(device)
        
        seg_mask = seg_mask * 2 - 1
        real_img = real_img * 2 - 1
        ########## BLUUUUUUUUUUUUUUUUUUURRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
        blurred_img = blur((real_img + 1) / 2.0) 
        blurred_img = blurred_img * 2 - 1 

        t = torch.randint(0, T, (real_img.size(0),), device=device).long() 
        x_t, noise = q_sample(real_img, t) 
        pred_noise = model(x_t, seg_mask.float(), blurred_img.float(), t)
        loss = diffusion_loss(pred_noise, noise, seg_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    l = total_loss / len(train_loader)
    print(f"Epoch {epoch} | Loss: {total_loss / len(train_loader):.4f}")
    
    import torchvision.utils as vutils

    sample_dir = "diffusion_t1N_all500" ##################################################################################
    os.makedirs(sample_dir, exist_ok=True)

    model.eval()
    val_total_loss = 0
    sampled_once = False

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    with torch.no_grad():
        for batch_idx, (real_img, seg_mask) in enumerate(val_loader):
            seg_mask, real_img = seg_mask.to(device), real_img.to(device)
    
            # Preprocess
            real_img = real_img * 2 - 1
            seg_mask = seg_mask * 2 - 1
            blurred_img = blur((real_img + 1) / 2.0)
            blurred_img = blurred_img * 2 - 1
    
            # --- Validation Loss ---
            t = torch.randint(0, T, (real_img.size(0),), device=device).long()
            x_t, noise = q_sample(real_img, t)
            pred_noise = model(x_t, seg_mask.float(), blurred_img.float(), t)
            val_total_loss+= diffusion_loss(pred_noise, noise, seg_mask).item()
            # val_total_loss += F.mse_loss(pred_noise, noise).item()

            x_sample = torch.randn_like(real_img).to(device)
            for t_inv in reversed(range(0, T)):
                t_vis = torch.full((seg_mask.size(0),), t_inv, device=device, dtype=torch.long)
                pred_noise = model(x_sample, seg_mask.float(), blurred_img.float(), t_vis)
                alpha = alphas[t_vis][:, None, None, None]
                beta = betas[t_vis][:, None, None, None]
                x_sample = (1 / torch.sqrt(alpha)) * (x_sample - (beta / torch.sqrt(1 - alphas_cumprod[t_vis])[:,None, None, None]) * pred_noise)
                if t_inv > 0:
                    sigma = torch.sqrt(betas[t_vis])[:, None, None, None]
                    x_sample += sigma * torch.randn_like(x_sample)
            x_sample = (x_sample + 1) / 2.0
            real_img_eval = (real_img + 1) / 2.0

            # --- Accumulate metrics ---
            psnr_b, ssim_b = compute_metrics(real_img_eval, x_sample)
            total_psnr += psnr_b * real_img.size(0)
            total_ssim += ssim_b * real_img.size(0)
            count += real_img.size(0)

            # Visualization
            if batch_idx == 0:
                vis_real_img = (blurred_img + 1) / 2.0
                seg_mask_vis = (seg_mask + 1) / 2.0
                display_grid = torch.cat([seg_mask_vis[:8], vis_real_img[:8], x_sample[:8]], dim=0)
                grid = vutils.make_grid(display_grid, nrow=8, padding=2)
                vutils.save_image(grid, os.path.join(sample_dir, f"epoch_{epoch:03d}.png"))
                print(f"🖼️ Saved sample image grid at epoch {epoch} → {sample_dir}/epoch_{epoch:03d}.png")
                
    val_loss = val_total_loss / len(val_loader)
    psnr_val = total_psnr / count
    ssim_val = total_ssim / count
    print(f"📉 Val Loss: {val_loss:.4f}")
    
    print(f"📉 Epoch {epoch} | Train Loss: {l:.4f} | Val Loss: {val_loss:.4f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f}")
    
    # Append to JSON log
    metrics_log.append({
        "epoch": epoch,
        "train_loss": round(l, 4),
        "val_loss": round(val_loss, 4),
        "psnr": round(psnr_val, 2),
        "ssim": round(ssim_val, 4)
    })
    
    # Write JSON log
    with open("t1n_all500.json", "w") as f:
        json.dump(metrics_log, f, indent=4)


    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0  # reset counter
        torch.save(model.state_dict(), os.path.join(save_dir, "best_generator_t1N_all500.pth")) #########################################################
        print(f"✅ Saved new best model based on Val Loss: {val_loss:.4f}")
    else:
        epochs_without_improvement += 1
        print(f"⏳ No improvement for {epochs_without_improvement} epoch(s)")
    
    # Always save latest model
    torch.save(model.state_dict(), os.path.join(save_dir, "best_generator_t1N_all500_finalepoch.pth"))   ##################################################     
    
    # Trigger early stop
    if epochs_without_improvement >= early_stopping_patience:
        print(f"🛑 Early stopping triggered after {epoch+1} epochs.")
        break  # Exit training loop

    print(f"✅ Saved new best model with final epoch")