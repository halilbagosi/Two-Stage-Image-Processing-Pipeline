# 🎬 Stage 2: DSLR to Cinematic Film Enhancement

## Overview

This stage transforms DSLR-quality photos into professional cinematic film looks with Hollywood-style color grading.

```
DSLR Photo  →  [Stage 2 Model]  →  Cinematic Film Photo
```

## What It Does

- 🎨 Teal/orange cinematic color grading
- 📈 S-curve contrast enhancement
- 🌅 Shadow lift and highlight roll-off
- 🎥 Film-like tonal transitions
- ✨ Professional depth and dimension

## Architecture

- **Model**: 3D LUT-based Film Enhancement Network
- **Parameters**: ~142K (very efficient!)
- **Input**: DSLR-enhanced photos (from Stage 1)
- **Output**: Cinematic film photos

## Film Styles

Three built-in styles:

### 1. Cinematic (Default) 🎬
- Teal shadows, orange highlights
- Modern blockbuster look
- Hollywood color palette

### 2. Vintage 📼
- Faded colors
- Classic film aesthetic
- Warm nostalgic tone

### 3. Modern 🎯
- High contrast
- Saturated colors
- Contemporary look

## Training

### Quick Start

**For M4 users** (30-40% faster!):
```bash
# From project root
cd code/Stage2-Film-Enhancement

# Activate virtual environment
source ../../venv/bin/activate

# Start training with M4 optimization
python3 train.py --config configs/stage2_config_m4.yaml
```

**For M1/M2/M3 users**:
```bash
python3 train.py --config configs/stage2_config.yaml
```

### Timeline

**M4 Chip**:
- **Epochs**: 80
- **Time per epoch**: 10-14 minutes
- **Total time**: ~14-16 hours ⚡

**M1/M2/M3**:
- **Epochs**: 80
- **Time per epoch**: 15-20 minutes
- **Total time**: ~20-25 hours

### Configuration

Edit configs to customize:
- `stage2_config_m4.yaml` - M4 optimized (batch_size: 24)
- `stage2_config.yaml` - Standard (batch_size: 16)

Options:
- Film style (cinematic/vintage/modern)
- Color intensity
- Contrast boost
- Saturation levels

## Dataset

Uses **procedural target generation**:
- Takes DSLR images as input
- Applies film transforms to create targets
- No additional dataset needed!

**Input**: Same DSLR images from Stage 1 dataset

## Results

After training, expect:
- **PSNR**: 26-30 dB
- **SSIM**: 0.90-0.95
- **Visual**: Professional cinematic film look

## Files

### Core Training
- `train.py` - Main training script
- `test_stage2_setup.py` - Setup verification

### Model Architecture
- `models/stage2_film.py` - LUT & CNN models
- `models/blocks.py` - Shared building blocks

### Data & Loss
- `data/stage2_dataset.py` - Data loader
- `data/film_transforms.py` - Film look generation
- `losses/film_loss.py` - Film-specific losses

### Configuration
- `configs/stage2_config_m4.yaml` - M4 optimized
- `configs/stage2_config.yaml` - Standard

## Parallel Training

**Pro Tip**: You can train Stage 1 and Stage 2 simultaneously!
- Both fit in 16GB RAM
- Saves ~6 days total time
- Just open two terminals

## Next Step

After Stage 2 training completes, use **Combined-Inference** to process images end-to-end!

---

**Stage 2 of 2** - DSLR to Cinematic Film Enhancement

