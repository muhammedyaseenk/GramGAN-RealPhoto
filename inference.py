#!/usr/bin/env python3
"""
Gram-GAN Inference Script
-------------------------------------
Loads a trained RRDBNet generator and performs 4x super-resolution
on a single image or all images in a folder.

Usage (CLI):
  python gramgan_infer.py --weights ./gramgan_G_best.pth --input ./test_images --output ./sr_results

Usage (Quick test with hardcoded config):
  Just run `python gramgan_infer.py` (will use the default config at the bottom)
"""

import os
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

import sys
print(sys.executable)


# ==========================================================
# 1. Generator Architecture (RRDBNet)
# ==========================================================
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.scale = 0.2

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + x5 * self.scale


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)
        self.scale = 0.2

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.scale


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
        self.upsampler = nn.PixelShuffle(2)
        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upsampler(self.upconv1(fea)))
        fea = self.lrelu(self.upsampler(self.upconv2(fea)))
        out = self.conv_last(self.lrelu(self.hr_conv(fea)))
        return out


# ==========================================================
# 2. Load generator weights
# ==========================================================
def load_generator(weights_path, device):
    model = RRDBNet(scale=4)
    ckpt = torch.load(weights_path, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    return model


# ==========================================================
# 3. Inference helper
# ==========================================================
def run_inference(model, input_path, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Handle single file or folder
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                 if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for f in tqdm(files, desc="Super-resolving"):
        img = Image.open(f).convert("RGB")
        lr = to_tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            sr = model(lr)
            sr = torch.clamp(sr, 0, 1)

        sr_img = to_pil(sr.squeeze(0).cpu())
        save_path = os.path.join(output_dir, os.path.basename(f))
        sr_img.save(save_path)

    print(f"Saved results to: {output_dir}")


# ==========================================================
# 4. CLI / Direct Execution
# ==========================================================
def main(args=None):
    device = torch.device(args.device if args else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model
    model = load_generator(args.weights, device)
    run_inference(model, args.input, args.output, device)


if __name__ == "__main__":
    # Define both CLI and default config modes
    parser = argparse.ArgumentParser(description="Gram-GAN inference script")
    parser.add_argument("--weights", type=str,
                        default=r"D:\Projects\PYTHONPROJECTS\output_test_run\gramgan_cpu_test.pth",
                        help="Path to trained generator .pth file")
    parser.add_argument("--input", type=str,
                        default=r"D:\Projects\PYTHONPROJECTS\lowres_images\img_cats_4.jpg",
                        help="Path to LR image or folder")
    parser.add_argument("--output", type=str,
                        default=r"D:\Projects\PYTHONPROJECTS\test_output",
                        help="Output folder for SR results")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args)
