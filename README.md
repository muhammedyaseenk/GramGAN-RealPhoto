# GramGAN-RealPhoto
Gram-GAN bridges pixel-based and GAN SR methods, generating sharp and realistic textures by explicitly matching generated patches to real HR textures via a Gram-based patch database. It ensures stable training, better generalization, and texture-aware outputs—ideal for satellite, medical, art, and fashion super-resolution tasks.


## Overview: Gram-GAN - Texture-Aware Super-Resolution

**Gram-GAN** is a super-resolution GAN designed to generate **sharp and realistic textures** by explicitly guiding the generator with a **Gram-based patch database** derived from high-resolution images. It bridges the gap between pixel-based SR methods and GAN-based SR, providing perceptually realistic results while stabilizing adversarial training.

---

## 🧭 Why Gram-GAN?

Traditional super-resolution models often face a trade-off:

| Type                        | Example       | Problem                                    |
| --------------------------- | ------------- | ----------------------------------------- |
| Pixel-based (MSE/L1)        | SRCNN, EDSR   | Sharp edges but smooth textures           |
| Adversarial (GAN-based)     | SRGAN, ESRGAN | Sharp but unrealistic/noisy textures      |

Gram-GAN **bridges these extremes**:  
✅ Sharp **and** realistic textures  
✅ Stable and controlled GAN training  
✅ Explicitly enforces texture realism using Gram matrices of real HR patches

---

## 🎯 Key Features

| Feature                 | ESRGAN / SRGAN         | Gram-GAN                                           |
| ----------------------- | -------------------- | ------------------------------------------------- |
| Texture supervision     | Implicit (adversarial) | **Explicit**, patch Gram matching                 |
| Texture reference       | None                  | **Patch database of real HR textures**           |
| Perceptual losses       | VGG only              | VGG + Discriminator Perceptual + Texture loss    |
| Adversarial type        | Standard Relativistic GAN | Relativistic + Texture-informed training       |
| Region masking          | None                  | Texture-aware masks focusing on details          |
| Texture diversity       | Depends on generator  | Driven by real HR patch library                  |

**TL;DR:** Gram-GAN doesn’t just hope to learn textures; it **shows the model what “real” texture looks like**.

---

## ⚙️ How It Works

The core mechanism is the **Gram Patch Texture Loss (L_PT)**:

- Each generated patch is compared to the closest real HR patch in **Gram-space**.
- Ensures generated textures match **real-world texture statistics**.
- Training behaves like learning from a **real texture dictionary**, not hallucinations.

---

## 🧩 Benefits in Practice

| Benefit                        | Description                                                             |
| ------------------------------- | ----------------------------------------------------------------------- |
| More natural textures           | Follows realistic patterns for wood, skin, fabric, grass, etc.          |
| Stable adversarial training     | Patch-based Gram supervision reduces mode collapse                       |
| Better generalization           | Performs well on unseen domains                                         |
| Modular training                | Patch database can be updated independently                              |
| Texture awareness               | Masking focuses GAN attention on critical regions                        |

---

## 🏗️ Ideal Use Cases

- **Aerial & Satellite Imaging:** Terrain, crops, roads  
- **Medical Imaging:** MRI, X-ray, or micro-texture enhancement  
- **Cultural Heritage / Art Restoration:** Paintings, documents  
- **Face & Fashion Super-Resolution:** Skin, hair, clothing details  
- **Industrial Inspection:** Microscope or camera captures of surfaces  

> Use Gram-GAN whenever **visual texture fidelity** matters more than just pixel-level accuracy.

---

## ⚔️ Limitations

- Pure numerical accuracy tasks (PSNR benchmarks) may not benefit  
- Requires sufficient HR patch data for diversity  
- Training overhead due to patch database; inference remains fast

---

## 🔮 Research Vision

Gram-GAN blends:

- **GANs** for realism  
- **Style transfer** for Gram-based texture representation  
- **Database retrieval** for local real examples  

It represents an early form of **texture-aware generative learning** and opens avenues for **patch-guided generation** or **texture memory GANs**.

---

## 🧠 Summary

Gram-GAN provides:

- Realistic, diverse, and high-frequency textures  
- Explicit patch-based supervision for stable GAN training  
- Modular and extendable patch database for multiple domains  

---

# Gram-GAN Implementation: GrahamGAN-RealPhoto

A PyTorch implementation of Gram-GAN for 4x image super-resolution based on Gram matrix and discriminator perceptual loss.

## Overview

GrahamGAN enhances low-resolution images using:

- **RRDB Generator:** 23 Residual-in-Residual Dense Blocks (ESRGAN backbone)  
- **Gram Matrix Supervision:** Patch-wise texture matching  
- **Discriminator Perceptual Loss:** Feature matching from discriminator layers  
- **Relativistic Average GAN:** Region-masked adversarial training  

---

## Features

- 4x super-resolution (48x48 → 192x192 patches)  
- Patch candidate database with Faiss acceleration  
- Live training with streaming logs  
- Robust error handling and diagnostics  
- Configurable training parameters  

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd GrahamGAN

# Install dependencies
pip install torch torchvision tqdm pillow numpy
pip install faiss-cpu  # or faiss-gpu for GPU acceleration
pip install lpips      # optional, for evaluation
````

---

## Quick Start

### 1. Prepare Training Data

```python
python collect_images.py
```

### 2. Train the Model

```python
# Quick test run (50 iterations)
python trainer.py

# Full training
python graham_gan_model.py --hr_dir ./your_images --batch_size 8 --total_iters 600000
```

---

## Project Structure

```
GrahamGAN/
├── graham_gan_model.py
├── trainer.py
├── data_loader.py
├── collect_images.py
├── downloaded_images/
├── graham_model/
│   ├── gramgan_G_best.pth
│   └── gramgan_G_best_D.pth
└── README.md
```

---

## Model Architecture

### Generator (RRDBNet)

* Input: 3-channel LR image
* Backbone: 23 RRDB blocks (nf=64, gc=32)
* Upsampling: 2x PixelShuffle layers for 4x scale
* Output: 3-channel HR image

### Discriminator

* VGG-style, 11 conv layers
* Extracts conv#5 and conv#11 for perceptual loss
* Outputs real/fake logits

### Loss Components

```
L_total = η₁L_PT + η₂L_DP + η₃L_P + η₄L_G + η₅L_C
```

* **L_PT:** Patch texture loss (Gram matrix matching)
* **L_DP:** Discriminator perceptual loss
* **L_P:** VGG perceptual loss
* **L_G:** Relativistic average GAN loss
* **L_C:** Content loss (L1)

---

## Training Configuration

```python
default_configure = {
    "hr_dir": "./data/DIV2K_train_HR",
    "batch_size": 16,
    "base_lr": 2e-4,
    "max_patch_db": 100000,
    "total_iters": 200000,
    "output_dir": "./output_gram_gan"
}
```

---

## Command Line Usage

```bash
# Basic training
python graham_gan_model.py --hr_dir ./images --batch_size 8

# Full configuration
python graham_gan_model.py \
    --hr_dir ./DIV2K_train_HR \
    --batch_size 16 \
    --base_lr 2e-4 \
    --total_iters 600000 \
    --max_patch_db 200000 \
    --save_path ./checkpoints/
```

---

## Key Features

* Patch candidate database with 4x4 patches from training images
* Faiss indexing for fast nearest-neighbor search
* Texture-aware region masking
* Live streaming logs and diagnostics
* Configurable LR scheduling and checkpointing

---

## Performance

* Test run: ~2–3 minutes (50 iterations)
* Full training: ~24–48 hours (600k iterations)
* GPU: 8GB+ recommended for batch_size=8
* RAM: 16GB+ for patch database

---

## Troubleshooting

* **CUDA out of memory:** Reduce batch_size
* **Slow training:** Enable Faiss GPU indexing
* **Poor results:** Increase max_patch_db
* **Import errors:** Install missing dependencies

Logs include loss components, learning rate schedules, patch DB stats, and diagnostics.

---

## Citation

```
Song et al., "Gram-GAN: Image Super-Resolution Based on Gram Matrix 
and Discriminator Perceptual Loss", Sensors 2023.
```

---

## License

This implementation is for **research and educational purposes**.
