# 🎬 Smartphone-DSLR Image Pipeline

AI-powered two-stage pipeline that transforms smartphone photos into professional DSLR-quality images with cinematic film aesthetics.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Optimized-000000.svg)

---

## 👥 Group Members & Roles

- **Halil Bagosi**: Lead Developer - Pipeline Architecture, Apple Silicon Optimization, and Training Engine.
- **Endi Cikalleshi**: AI Researcher - Dataset Analysis, Metric Validation, and Documentation.

---

## 🎯 Overview & AI Approach

This project implements a **Model-Based Learning Agent** designed for high-fidelity image enhancement. The agent uses a two-stage deep learning pipeline to transform smartphone photos into professional DSLR-quality images:

```
Smartphone Photo
    ↓
[Stage 1: Smartphone → DSLR]
    ↓
DSLR-Quality Photo
    ↓
[Stage 2: DSLR → Cinematic Film]
    ↓
Professional Cinematic Result 🎬
```

### Features

- ✨ **Two-Stage Enhancement**: Smartphone → DSLR → Cinematic Film
- 🚀 **Apple Silicon Optimized**: M1/M2/M3/M4 with MPS acceleration
- 🎨 **Multiple Film Styles**: Cinematic, Vintage, Modern
- ⚡ **Fast Inference**: ~0.1 seconds per image
- 📦 **Production Ready**: Complete training & inference pipeline

---

## 📊 Results

### Stage 1: Smartphone → DSLR Enhancement
- Improves shadows, highlights, saturation, tone, clarity
- Matches DSLR sensor characteristics
- Preserves natural look

### Stage 2: DSLR → Cinematic Film
- Professional color grading (teal/orange)
- Film-like tonal curves
- Enhanced depth and dimension

---

## 🏗️ Architecture

### Stage 1: Lightweight U-Net
- **Model**: U-Net with attention mechanisms
- **Parameters**: ~1.1M
- **Training**: 55 epochs with best model at epoch 25.
- **Loss**: L1 + SSIM

### Stage 2: Film LUT Network
- **Model**: 3D LUT-based enhancement
- **Parameters**: ~142K
- **Training**: 100 epochs
- **Loss**: L1 + Style + Color consistency

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Apple Silicon Mac (M1/M2/M3/M4) or CUDA GPU
- 16GB+ RAM recommended

### Installation & Quick Start

The easiest way to get started is using the `quick_start.py` script:

```bash
# Clone the repository
git clone https://github.com/yourusername/smartphone-dslr-pipeline.git
cd smartphone-dslr-pipeline

# Run the auto-setup and launch the GUI
python3 quick_start.py
```

Alternatively, you can install dependencies manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install consolidated dependencies
pip install -r requirements.txt
```

### 🖼️ Enhanced Image Support

This version now supports all major image formats out of the box:
- **Standard**: JPEG, JPG, PNG
- **RAW**: DNG
- **Modern**: HEIF / HEIC (iPhone photos)
- **Automatic Device Detection**: Works on Apple Silicon (MPS), NVIDIA GPUs (CUDA), and CPU.

### Dataset

Download the [DPED (DSLR Photo Enhancement Dataset)](http://people.ee.ethz.ch/~ihnatova/#dataset):

```bash
# Place dataset in:
datasets/dped/
├── iphone/
│   ├── train/
│   │   ├── patches/
│   └── test/
├── sony/
└── blackberry/
```

---

## 🎓 Training

### Stage 1: Smartphone → DSLR (Exposure-Only Model)

```bash
cd code/Stage1-Smartphone-to-DSLR
python3 train.py --config configs/exposure.yaml
```

**What it does**: Adjusts exposure, highlights, shadows, and contrast while preserving colors and details.

**Timeline**: ~4-5 days (150 epochs) on Apple Silicon

**Timeline**: ~10 days on M1 Pro

### Stage 2: DSLR → Cinematic Film

**For M4 users** (recommended):
```bash
cd code/Stage2-Film-Enhancement
python3 train.py --config configs/stage2_config_m4.yaml
```
**Timeline**: ~14-16 hours on M4

**For M1/M2/M3 users**:
```bash
cd code/Stage2-Film-Enhancement
python3 train.py --config configs/stage2_config.yaml
```
**Timeline**: ~20-25 hours

### Parallel Training

Both stages can run simultaneously:
```bash
# Terminal 1: Stage 1
cd code/Stage1-Smartphone-to-DSLR
python3 train.py --config configs/exposure.yaml

# Terminal 2: Stage 2
cd code/Stage2-Film-Enhancement
python3 train.py --config configs/stage2_config_m4.yaml
```

---

## 🎬 Inference

### Single Image

```bash
cd code/Combined-Inference

# Both stages
python3 inference.py \
  --stage1 checkpoints/stage1_production/best_model.pth \
  --stage2 checkpoints/stage2_m4_optimized/best_model.pth \
  --input photo.jpg \
  --output ./results

# Stage 1 only
python3 inference.py \
  --stage1 checkpoints/stage1_production/best_model.pth \
  --input photo.jpg \
  --output ./results
```

### Batch Processing

```bash
python3 inference.py \
  --stage1 checkpoints/stage1_production/best_model.pth \
  --stage2 checkpoints/stage2_m4_optimized/best_model.pth \
  --input /path/to/photos/folder \
  --output ./batch_results
```

### Python API

```python
from inference import SmartphoneToFilmPipeline

# Initialize pipeline
pipeline = SmartphoneToFilmPipeline(
    stage1_checkpoint='checkpoints/stage1_production/best_model.pth',
    stage2_checkpoint='checkpoints/stage2_m4_optimized/best_model.pth',
    device='mps'
)

# Process image
results = pipeline.process('photo.jpg', output_dir='./results')
```

---

## 📚 Documentation
Comprehensive documentation for the project is available in the master guide:

- **[PRESENTATION_MASTER_GUIDE.md](PRESENTATION_MASTER_GUIDE.md)**: The definitive technical reference for architectures, training, optimization, and success metrics.
- **[PROJECT_FILES_GUIDE.md](PROJECT_FILES_GUIDE.md)**: Details of every necessary file in the repository.

---

## 🛠️ Project Structure
**Organized into 3 clear folders for developer focus:**

```
Smartphone-DSLR-Image-Pipeline/
├── code/
│   ├── Stage1-Smartphone-to-DSLR/    📱 Smartphone → DSLR
│   │   ├── train.py                  # Training engine
│   │   └── models/                   # DPED & ZRR architectures
│   │
│   ├── Stage2-Film-Enhancement/       🎬 DSLR → Film
│   │   ├── train.py                  # Training engine
│   │   └── models/                   # Film LUT networks
│   │
│   └── Combined-Inference/            🎯 End-to-End Production
│       ├── inference.py              # Master inference logic (Tiled)
│       ├── gui.py                    # Gradio GUI
│       └── gui_streamlit.py          # Streamlit GUI
│
├── visual_metrics/                    # Generated performance analytics
├── PRESENTATION_MASTER_GUIDE.md       # Technical Reference (Report Source)
└── README.md                          # Repository Entry
```

---

## ⚡ Performance

### M4 Chip (Recommended)
- **Stage 1**: ~23 min/epoch, ~10 days total
- **Stage 2**: ~11 min/epoch, ~14-16 hours total
- **Inference**: ~0.06 sec/image

### M1 Pro
- **Stage 1**: ~23 min/epoch, ~10 days total
- **Stage 2**: ~17 min/epoch, ~20-25 hours total
- **Inference**: ~0.1 sec/image

### Memory Usage
- **Stage 1**: ~3-4GB
- **Stage 2**: ~2-3GB
- **Both parallel**: ~6-7GB

---

## 🎨 Film Styles

### Cinematic (Default)
- Teal/orange color grading
- S-curve contrast
- Hollywood blockbuster look

### Vintage
- Faded colors
- Reduced contrast
- Classic film aesthetic

### Modern
- High contrast
- Saturated colors
- Contemporary look

Train different styles by editing `configs/stage2_config.yaml`:
```yaml
dataset:
  film_style: 'cinematic'  # or 'vintage', 'modern'
```

---

## 📊 Results & Metrics

### Stage 1 (Smartphone → DSLR)
- **PSNR**: 24-28 dB
- **SSIM**: 0.85-0.90
- **Visual**: Natural DSLR enhancement

### Stage 2 (DSLR → Film)
- **PSNR**: 26-30 dB
- **SSIM**: 0.90-0.95
- **Visual**: Professional cinematic look

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [DPED (DSLR Photo Enhancement Dataset)](http://people.ee.ethz.ch/~ihnatova/#dataset)
- **Architecture Inspiration**: U-Net, 3D LUT-based enhancement
- **Hardware**: Optimized for Apple Silicon

---

## 📧 Contact

For questions or collaborations, please open an issue.

---

## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Transform your smartphone photos into cinematic masterpieces!** 🎬📸✨

