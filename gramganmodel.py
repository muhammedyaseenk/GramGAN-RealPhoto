#!/usr/bin/env python3
"""
Gram-GAN exact implementation (PyTorch) - Refactored/robust version
Integrated corrections:
 - Deterministic discriminator feature extraction (conv #5 and #11)
 - VGG19 feature extractor: ImageNet normalization + eval + freeze
 - Faiss query uses weighted combination of Gram(gt) and Gram(est)
 - Defensive device `.to(device)` for Gram tensors before concatenation
 - Clearer RaGAN loss variable names
 - Optional Kaiming initialization helper called on G and D
 - Minor robustness fixes retained from original (affine fallback, reshape gram, patches)
"""
import os
import math
import random
import argparse
import inspect
from typing import List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm

# Optional evaluation libs
try:
    import lpips
except Exception:
    lpips = None

# torchvision interpolation import (may not exist on older versions)
try:
    from torchvision.transforms import InterpolationMode
except Exception:
    InterpolationMode = None

# Optional Faiss
try:
    import faiss
except Exception:
    faiss = None

# -----------------------------
# Helper: safe argparse add
# -----------------------------
def _add_arg_if_missing(parser: argparse.ArgumentParser, *option_strings, **kwargs):
    """Add an argument only if none of the option_strings already exist on the parser.
    If it exists and a 'default' is supplied, update the existing action's default.
    """
    for opt in option_strings:
        if opt in parser._option_string_actions:
            # update default if provided
            if 'default' in kwargs:
                action = parser._option_string_actions[opt]
                action.default = kwargs['default']
            return
    parser.add_argument(*option_strings, **kwargs)


# ---------------------------------------------------------------------
# Utils: Gram matrix, patch extraction & robust affine transform
# ---------------------------------------------------------------------
def gram_matrix(p: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix G(p) = p^T p for a patch tensor.
       p: (C, H, W) or (B, C, H, W)
       returns: (C, C) or (B, C, C)
    """
    if p.dim() == 3:
        C, H, W = p.shape
        x = p.reshape(C, H * W)  # reshape handles non-contiguous tensors
        G = x @ x.t()
        return G
    elif p.dim() == 4:
        B, C, H, W = p.shape
        x = p.reshape(B, C, H * W)
        G = torch.bmm(x, x.transpose(1, 2))
        return G
    else:
        raise ValueError("Unsupported tensor shape for gram matrix")


def apply_small_affine(pil_img: Image.Image, lambda_aff: float = 0.003) -> Image.Image:
    """Apply a small random affine transform (robust to different torchvision versions)."""
    w, h = pil_img.size
    angle = float(torch.randn(1).item() * lambda_aff * 360.0)
    translate = (float(torch.randn(1).item() * lambda_aff * w),
                 float(torch.randn(1).item() * lambda_aff * h))
    scale = 1.0 + float(torch.randn(1).item() * lambda_aff)
    shear = float(torch.randn(1).item() * lambda_aff * 20.0)

    common_kwargs = dict(angle=angle, translate=translate, scale=scale, shear=shear)

    pil_interp = Image.BICUBIC

    try:
        sig = inspect.signature(TF.affine)
        params = sig.parameters
    except Exception:
        params = {}

    # Try newer 'interpolation' kwarg
    if "interpolation" in params and InterpolationMode is not None:
        try:
            return TF.affine(pil_img, interpolation=InterpolationMode.BICUBIC, **common_kwargs)
        except Exception:
            pass

    # Try older 'resample' kwarg
    if "resample" in params:
        try:
            return TF.affine(pil_img, resample=pil_interp, **common_kwargs)
        except Exception:
            pass

    # Fallback
    try:
        return TF.affine(pil_img, **common_kwargs)
    except Exception:
        try:
            # Approximate fallback: rotate + translate
            return pil_img.rotate(angle, resample=pil_interp)
        except Exception:
            return pil_img


def extract_patches_tensor(img: torch.Tensor, patch_size:int, stride:int=None) -> torch.Tensor:
    """Extract patches from a tensor image. img: (C,H,W) or (B,C,H,W).
       Returns: (N, C, patch_size, patch_size)
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    B, C, H, W = img.shape
    if stride is None:
        stride = patch_size
    patches = img.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(2,0,1,3,4).contiguous().view(-1, C, patch_size, patch_size)
    return patches  # (N, C, ph, pw)


# ----------------------------------------------------------------------------- 
# Dataset: DIV2K style loader + sliding crop to produce HR patches as paper
# ----------------------------------------------------------------------------- 
class DIV2KTrainDataset(Dataset):
    def __init__(self, hr_dir: str, hr_patch_size:int = 192, scale:int=4, augment=True):
        self.hr_dir = hr_dir
        self.files = sorted([os.path.join(hr_dir,f) for f in os.listdir(hr_dir)
                             if f.lower().endswith(('png','jpg','jpeg'))])
        self.hr_patch_size = hr_patch_size
        self.scale = scale
        self.lr_patch_size = hr_patch_size // scale
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return max(1, len(self.files)) * 60

    def __getitem__(self, idx):
        f = self.files[idx % len(self.files)]
        hr = Image.open(f).convert('RGB')
        # Ensure minimal size
        if hr.width < self.hr_patch_size or hr.height < self.hr_patch_size:
            hr = hr.resize((max(self.hr_patch_size, hr.width), max(self.hr_patch_size, hr.height)), Image.BICUBIC)
        x = random.randint(0, hr.width - self.hr_patch_size)
        y = random.randint(0, hr.height - self.hr_patch_size)
        hr_crop = hr.crop((x,y, x+self.hr_patch_size, y+self.hr_patch_size))
        lr_crop = hr_crop.resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)
        if self.augment:
            if random.random() < 0.5:
                hr_crop = TF.hflip(hr_crop); lr_crop = TF.hflip(lr_crop)
            if random.random() < 0.5:
                hr_crop = TF.vflip(hr_crop); lr_crop = TF.vflip(lr_crop)
            k = random.randint(0,3)
            if k:
                hr_crop = hr_crop.rotate(90*k); lr_crop = lr_crop.rotate(90*k)
        hr_t = self.to_tensor(hr_crop)
        lr_t = self.to_tensor(lr_crop)
        return lr_t, hr_t


# ----------------------------------------------------------------------------- 
# Patch Candidate DB (with optional Faiss) 
# ----------------------------------------------------------------------------- 
class PatchCandidateDB:
    def __init__(self, hr_dir: str, patch_size:int = 4, lambda_aff:float=0.003,
                 max_patches:int=100000, use_faiss:bool=True, topk:int=10):
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.lambda_aff = lambda_aff
        self.max_patches = max_patches
        self.use_faiss = use_faiss and (faiss is not None)
        self.topk = topk

        self.patches = []
        self.grams = []
        self.vecs = None
        self.faiss_index = None

        self._build_db()
        if self.use_faiss:
            self._build_faiss_index()

    def _build_db(self):
        files = sorted([os.path.join(self.hr_dir,f)
                        for f in os.listdir(self.hr_dir)
                        if f.lower().endswith(('png','jpg','jpeg'))])
        to_tensor = transforms.ToTensor()
        for f in files:
            im = Image.open(f).convert('RGB')
            w,h = im.size
            for _ in range(60):
                if len(self.patches) >= self.max_patches:
                    break
                x = random.randint(0, max(0,w-self.patch_size))
                y = random.randint(0, max(0,h-self.patch_size))
                p = im.crop((x,y,x+self.patch_size,y+self.patch_size))
                pt = to_tensor(p)
                self.patches.append(pt.cpu())
                self.grams.append(gram_matrix(pt).cpu())

                # downsample-upsample variant
                p_down = p.resize((max(1,self.patch_size//2), max(1,self.patch_size//2)), Image.BICUBIC)
                p_du = p_down.resize((self.patch_size,self.patch_size), Image.BICUBIC)
                pt2 = to_tensor(p_du)
                self.patches.append(pt2.cpu()); self.grams.append(gram_matrix(pt2).cpu())

                # affine variant
                p_aff = apply_small_affine(p, self.lambda_aff)
                pt3 = to_tensor(p_aff)
                self.patches.append(pt3.cpu()); self.grams.append(gram_matrix(pt3).cpu())

            if len(self.patches) >= self.max_patches:
                break
        print(f"[PatchDB] built {len(self.patches)} patches")

    def _build_faiss_index(self):
        if faiss is None:
            print("[PatchDB] faiss not available, skipping index build.")
            self.use_faiss = False
            return
        print("[PatchDB] Building Faiss index...")
        vecs = torch.stack([g.flatten() for g in self.grams], dim=0)
        self.vecs = vecs.numpy().astype('float32')
        dim = self.vecs.shape[1]
        index = faiss.IndexFlatL2(dim)
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("[PatchDB] Using Faiss GPU index")
        else:
            print("[PatchDB] Using Faiss CPU index")
        index.add(self.vecs)
        self.faiss_index = index
        print(f"[PatchDB] Indexed {len(self.patches)} patches with dim={dim}")

    def find_best_patch(self, G_gi: torch.Tensor, G_ei: torch.Tensor,
                        alpha:float=0.5, beta:float=0.5) -> torch.Tensor:
        """
        Return patch tensor (C,H,W) chosen by minimizing:
          alpha * ||G(p) - G_gi||^2 + beta * ||G(p) - G_ei||^2
        Uses Faiss to shortlist candidates if available (query uses weighted combo).
        """
        Ggi = G_gi.cpu()
        Gei = G_ei.cpu()
        if self.use_faiss and self.faiss_index is not None:
            # Query with weighted combination vector (closer to paper objective)
            q = (alpha * Ggi.flatten() + beta * Gei.flatten()).unsqueeze(0).numpy().astype('float32')
            D,I = self.faiss_index.search(q, min(self.topk, len(self.patches)))
            candidate_idx = I[0]
        else:
            candidate_idx = range(len(self.grams))

        best_score = float('inf'); best_idx = None
        for i in candidate_idx:
            Gp = self.grams[i]
            s = alpha*torch.sum((Gp - Ggi)**2).item() + beta*torch.sum((Gp - Gei)**2).item()
            if s < best_score:
                best_score = s; best_idx = i
        return self.patches[best_idx]


# --------------------------------------------------------------------------
# Generator (RRDBNet), Discriminator, VGG features, Losses and utilities
# --------------------------------------------------------------------------
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf+gc, gc, 3,1,1)
        self.conv3 = nn.Conv2d(nf+2*gc, gc, 3,1,1)
        self.conv4 = nn.Conv2d(nf+3*gc, gc, 3,1,1)
        self.conv5 = nn.Conv2d(nf+4*gc, nf, 3,1,1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.scale = 0.2

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x,x1),1)))
        x3 = self.lrelu(self.conv3(torch.cat((x,x1,x2),1)))
        x4 = self.lrelu(self.conv4(torch.cat((x,x1,x2,x3),1)))
        x5 = self.conv5(torch.cat((x,x1,x2,x3,x4),1))
        return x + x5 * self.scale

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf,gc)
        self.rdb2 = ResidualDenseBlock_5C(nf,gc)
        self.rdb3 = ResidualDenseBlock_5C(nf,gc)
        self.scale = 0.2

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.scale

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3,1,1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3,1,1)
        self.upconv1 = nn.Conv2d(nf, nf*4, 3,1,1)
        self.upconv2 = nn.Conv2d(nf, nf*4, 3,1,1)
        self.upsampler = nn.PixelShuffle(2)
        self.hr_conv = nn.Conv2d(nf, nf, 3,1,1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3,1,1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.scale = scale

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upsampler(self.upconv1(fea)))
        fea = self.lrelu(self.upsampler(self.upconv2(fea)))
        out = self.conv_last(self.lrelu(self.hr_conv(fea)))
        return out


class DiscriminatorVGGStyle(nn.Module):
    def __init__(self, in_nc=3):
        super().__init__()
        modules = []
        c = 64
        modules.append(nn.Conv2d(in_nc, c, 3,1,1)); modules.append(nn.LeakyReLU(0.2,True))
        modules.append(nn.Conv2d(c, c, 3,2,1)); modules.append(nn.LeakyReLU(0.2,True))
        modules.append(nn.Conv2d(c, c*2, 3,1,1)); modules.append(nn.LeakyReLU(0.2,True))
        modules.append(nn.Conv2d(c*2, c*2,3,2,1)); modules.append(nn.LeakyReLU(0.2,True))
        modules.append(nn.Conv2d(c*2, c*4,3,1,1)); modules.append(nn.LeakyReLU(0.2,True))
        modules.append(nn.Conv2d(c*4, c*4,3,2,1)); modules.append(nn.LeakyReLU(0.2,True))
        modules.append(nn.Conv2d(c*4, c*8,3,1,1)); modules.append(nn.LeakyReLU(0.2,True))
        modules.append(nn.Conv2d(c*8, c*8,3,2,1)); modules.append(nn.LeakyReLU(0.2,True))
        modules.append(nn.Conv2d(c*8, c*16,3,1,1)); modules.append(nn.LeakyReLU(0.2,True))
        modules.append(nn.Conv2d(c*16, c*16,3,2,1)); modules.append(nn.LeakyReLU(0.2,True))
        modules.append(nn.Conv2d(c*16, c*16,3,1,1)); modules.append(nn.LeakyReLU(0.2,True))
        self.features = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c*16, 100), nn.LeakyReLU(0.2,True), nn.Linear(100,1))

        # Deterministically pick conv #5 and conv #11 (1-based conv count)
        conv_count = 0
        self.feature_layer_indices = []  # indices in the sequential module where we will capture outputs
        for idx, m in enumerate(self.features):
            if isinstance(m, nn.Conv2d):
                conv_count += 1
                if conv_count in (5, 11):
                    self.feature_layer_indices.append(idx)
        if len(self.feature_layer_indices) != 2:
            raise RuntimeError("Mismatch locating conv feature indices for L_DP (expected 2)")

    def forward(self, x, return_feats:bool=False):
        feats = []
        current = x
        for idx, layer in enumerate(self.features):
            current = layer(current)
            if idx in self.feature_layer_indices:
                feats.append(current)
        out = self.pool(current)
        out = self.classifier(out)
        if return_feats:
            return out, feats
        else:
            return out


class VGG19Features(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.to(device)
        self.layer_ids = {'conv3_4':16, 'conv4_4':23, 'conv5_4':30}
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.vgg.eval()
        self.device = device
        # ImageNet normalization (expects input in [0,1])
        self.register_buffer('imagenet_mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('imagenet_std', torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, x):
        x = x.to(self.device)
        x_norm = (x - self.imagenet_mean) / self.imagenet_std
        outputs = {}
        current = x_norm
        for i, layer in enumerate(self.vgg):
            current = layer(current)
            if i == self.layer_ids['conv3_4']:
                outputs['conv3_4'] = current
            if i == self.layer_ids['conv4_4']:
                outputs['conv4_4'] = current
            if i == self.layer_ids['conv5_4']:
                outputs['conv5_4'] = current
            if i >= self.layer_ids['conv5_4']:
                break
        return outputs


# -----------------------------
# Loss helpers
# -----------------------------
def build_mask_from_image(hr: torch.Tensor, patch_size:int = 11, delta: float = 0.005):
    """Compute binary mask M_{i,j} robust to H,W not divisible by patch_size."""
    B,C,H,W = hr.shape
    stride = patch_size
    patches = hr.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    n_h = patches.size(2); n_w = patches.size(3)
    patches = patches.contiguous().view(B, C, n_h*n_w, patch_size, patch_size)
    patches_flat = patches.view(B, C, n_h*n_w, -1)
    stds = patches_flat.std(dim=-1).mean(dim=1)  # (B, n_h*n_w)
    mask = (stds >= delta).float().view(B, 1, n_h, n_w)
    mask_up = F.interpolate(mask, size=(H, W), mode='nearest')
    return mask_up


def ra_adversarial_losses(C_real_logits, C_fake_logits):
    # Relativistic average GAN losses (BCE with logits)
    mean_real = torch.mean(C_real_logits)
    mean_fake = torch.mean(C_fake_logits)
    bce = nn.BCEWithLogitsLoss()
    LD_real = bce(C_real_logits - mean_fake, torch.ones_like(C_real_logits))
    LD_fake = bce(C_fake_logits - mean_real, torch.zeros_like(C_fake_logits))
    LD = LD_real + LD_fake
    LG_real = bce(C_real_logits - mean_fake, torch.zeros_like(C_real_logits))
    LG_fake = bce(C_fake_logits - mean_real, torch.ones_like(C_fake_logits))
    LG = LG_real + LG_fake
    return LD, LG


class GramGANLoss:
    def __init__(self, device):
        self.device = device
        self.vgg_feats = VGG19Features(device).to(device)
        self.vgg_layer_weights = {'conv3_4':1/8.0, 'conv4_4':1/4.0, 'conv5_4':1/2.0}
        self.eta = {'L_PT':1.0, 'L_DP':1.0, 'L_P':1.0, 'L_G':0.005, 'L_C':1.0}
        self.l1 = nn.L1Loss()

    def perceptual_vgg(self, sr, gt):
        sr_feats = self.vgg_feats(sr)
        gt_feats = self.vgg_feats(gt)
        loss = 0.0
        for k, w in self.vgg_layer_weights.items():
            loss = loss + w * F.l1_loss(sr_feats[k], gt_feats[k])
        return loss

    def discriminator_perceptual(self, sr_dis_feats:List[torch.Tensor], gt_dis_feats:List[torch.Tensor]):
        loss = 0.0
        for s,g in zip(sr_dis_feats, gt_dis_feats):
            loss = loss + F.l1_loss(s, g)
        return loss

    def patch_texture_loss(self, Gram_e, Gram_pstar):
        return F.l1_loss(Gram_e, Gram_pstar)

    def content_loss(self, sr, gt):
        return F.l1_loss(sr, gt)


# --------------------------------------------------------------------------
# Training utilities
# --------------------------------------------------------------------------
def adjust_lr_warmup_cosine(optimizer, base_lr, cur_iter, period_iters=200_000, warmup_iters=1000):
    iter_within = cur_iter % period_iters
    if iter_within < warmup_iters:
        lr = base_lr * (iter_within / warmup_iters)
    else:
        t = (iter_within - warmup_iters) / (period_iters - warmup_iters)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr

def init_weights(net, init_type='kaiming'):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4).to(device)
    D = DiscriminatorVGGStyle(in_nc=3).to(device)

    # init
    init_weights(G, 'kaiming')
    init_weights(D, 'kaiming')

    dataset = DIV2KTrainDataset(args.hr_dir, hr_patch_size=192, scale=4, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    patch_db = PatchCandidateDB(args.hr_dir,
                                patch_size=4,
                                lambda_aff=0.003,
                                max_patches=args.max_patch_db,
                                use_faiss=True)

    optG = torch.optim.Adam(G.parameters(), lr=args.base_lr, betas=(0.9,0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.base_lr, betas=(0.9,0.999))
    losses = GramGANLoss(device)

    cur_iter = 0
    pbar = tqdm(total=args.total_iters)
    while cur_iter < args.total_iters:
        for lr, hr in loader:
            if cur_iter >= args.total_iters: break
            lr = lr.to(device); hr = hr.to(device)
            sr = G(lr)
            mask = build_mask_from_image(hr, patch_size=11, delta=0.005).to(device)
            hr_masked = hr * mask
            sr_masked = sr * mask

            D_real_logits, D_real_feats = D(hr_masked, return_feats=True)
            D_fake_logits, D_fake_feats = D(sr_masked.detach(), return_feats=True)

            LD, _ = ra_adversarial_losses(D_real_logits, D_fake_logits)
            optD.zero_grad(); LD.backward(); optD.step()

            D_real_logits2, D_real_feats2 = D(hr_masked, return_feats=True)
            D_fake_logits2, D_fake_feats2 = D(sr_masked, return_feats=True)

            batch_size = sr.size(0)
            Gram_e_list = []
            Gram_pstar_list = []
            for b in range(batch_size):
                _,C,H,W = hr.shape
                ph = 4; pw = 4
                x0 = (W - pw)//2; y0 = (H - ph)//2
                ei = sr[b:b+1, :, y0:y0+ph, x0:x0+pw].squeeze(0).detach().cpu()
                gi = hr[b:b+1, :, y0:y0+ph, x0:x0+pw].squeeze(0).detach().cpu()
                Ggi = gram_matrix(gi)
                Gei = gram_matrix(ei)
                p_star = patch_db.find_best_patch(Ggi, Gei, alpha=0.5, beta=0.5)
                Gpstar = gram_matrix(p_star.to(device))
                Gram_e_list.append(Gei.unsqueeze(0).to(device))
                Gram_pstar_list.append(Gpstar.unsqueeze(0).to(device))

            Gram_e = torch.cat(Gram_e_list, dim=0)
            Gram_ps = torch.cat(Gram_pstar_list, dim=0)

            L_PT = losses.patch_texture_loss(Gram_e, Gram_ps)
            L_DP = losses.discriminator_perceptual(D_fake_feats2, D_real_feats2)
            L_P = losses.perceptual_vgg(sr, hr)
            L_C = losses.content_loss(sr, hr)
            _, LG = ra_adversarial_losses(D_real_logits2, D_fake_logits2)

            total_G = args.eta['L_PT']*L_PT + args.eta['L_DP']*L_DP + args.eta['L_P']*L_P + args.eta['L_G']*LG + args.eta['L_C']*L_C
            optG.zero_grad()
            total_G.backward()
            optG.step()

            adjust_lr_warmup_cosine(optG, args.base_lr, cur_iter, period_iters=200_000, warmup_iters=1000)
            adjust_lr_warmup_cosine(optD, args.base_lr, cur_iter, period_iters=200_000, warmup_iters=1000)

            if cur_iter % args.log_interval == 0:
                tqdm.write(f"Iter {cur_iter} | LD {LD.item():.6f} | LG {LG.item():.6f} | L_PT {L_PT.item():.6f} | L_DP {L_DP.item():.6f} | L_P {L_P.item():.6f} | L_C {L_C.item():.6f}")
            cur_iter += 1
            pbar.update(1)
            if cur_iter >= args.total_iters:
                break

    pbar.close()

    # handle save_path: if user passed a directory, turn into a filename inside it
    save_path = args.save_path or "gramgan_G.pth"
    if os.path.isdir(save_path):
        # append default filename with timestamp
        fname = f"gramgan_G_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        save_path = os.path.join(save_path, fname)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    torch.save(G.state_dict(), save_path)
    try:
        d_save = save_path.replace(".pth", "_D.pth")
        torch.save(D.state_dict(), d_save)
    except Exception:
        # best-effort for D checkpoint
        pass

    print(f"Training finished. Generator saved to: {save_path}")


# --------------------------------------------------------------------------
# CLI / parse args (robust, no duplicate add)
# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    _add_arg_if_missing(parser, '--hr_dir', type=str, required=True, help="DIV2K_train_HR folder or your images folder")
    _add_arg_if_missing(parser, '--batch_size', type=int, default=8)
    _add_arg_if_missing(parser, '--base_lr', type=float, default=1e-4)
    _add_arg_if_missing(parser, '--total_iters', type=int, default=600000)
    _add_arg_if_missing(parser, '--max_patch_db', type=int, default=200000)
    _add_arg_if_missing(parser, '--log_interval', type=int, default=500)

    # save_path: can be a file path or a directory
    default_save_path = r"gramgan_G_best.pth"
    _add_arg_if_missing(parser, '--save_path', type=str, default=default_save_path,
                        help="File path to save generator checkpoint (.pth) or a directory to place the file in.")

    _add_arg_if_missing(parser, '--eta_L_PT', type=float, default=1.0)
    _add_arg_if_missing(parser, '--eta_L_DP', type=float, default=1.0)
    _add_arg_if_missing(parser, '--eta_L_P', type=float, default=1.0)
    _add_arg_if_missing(parser, '--eta_L_G', type=float, default=0.005)
    _add_arg_if_missing(parser, '--eta_L_C', type=float, default=1.0)

    args = parser.parse_args()
    args.eta = {
        'L_PT': args.eta_L_PT,
        'L_DP': args.eta_L_DP,
        'L_P': args.eta_L_P,
        'L_G': args.eta_L_G,
        'L_C': args.eta_L_C
    }
    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isdir(args.hr_dir):
        raise SystemExit("Set --hr_dir to a folder containing your training images (png/jpg/jpeg).")
    train(args)
