# 🎬 Combined Inference: End-to-End Pipeline

## Overview

This combines both Stage 1 and Stage 2 for complete smartphone-to-cinematic transformation!

```
Smartphone Photo
    ↓
[Stage 1: Smartphone → DSLR]
    ↓
DSLR-Enhanced Photo
    ↓
[Stage 2: DSLR → Cinematic Film]
    ↓
Final Cinematic Result 🎬
```

## Usage

### Single Image

```bash
# From project root
cd code/Combined-Inference

# Activate virtual environment
source ../../venv/bin/activate

# Process one image
python3 inference.py \
  --stage1 ../Stage1-Smartphone-to-DSLR/checkpoints/stage1_production/best_model.pth \
  --stage2 ../Stage2-Film-Enhancement/checkpoints/stage2_m4_optimized/best_model.pth \
  --input /path/to/photo.jpg \
  --output ./results
```

### Batch Processing

```bash
# Process entire folder
python3 inference.py \
  --stage1 ../Stage1-Smartphone-to-DSLR/checkpoints/stage1_production/best_model.pth \
  --stage2 ../Stage2-Film-Enhancement/checkpoints/stage2_m4_optimized/best_model.pth \
  --input /path/to/photos/folder \
  --output ./batch_results
```

### Stage 1 Only

If you just want DSLR enhancement without film look:

```bash
python3 inference.py \
  --stage1 ../Stage1-Smartphone-to-DSLR/checkpoints/stage1_production/best_model.pth \
  --input photo.jpg \
  --output ./results
```

## Output Files

For each input image `photo.jpg`, you'll get:

```
results/
├── photo_0_input.png       # Original smartphone
├── photo_1_dslr.png        # After Stage 1
├── photo_2_film.png        # After Stage 2
└── photo_comparison.png    # Side-by-side
```

## Performance

### Speed
- **Single image (224x224)**: ~0.1 seconds
- **Single image (1024x1024)**: ~0.5 seconds
- **4K image**: ~2-3 seconds

### Memory
- **Stage 1 only**: ~300MB
- **Both stages**: ~500MB

## Python API

```python
from inference import SmartphoneToFilmPipeline

# Initialize
pipeline = SmartphoneToFilmPipeline(
    stage1_checkpoint='../Stage1-Smartphone-to-DSLR/checkpoints/stage1_production/best_model.pth',
    stage2_checkpoint='../Stage2-Film-Enhancement/checkpoints/stage2_m4_optimized/best_model.pth',
    device='mps'
)

# Process
results = pipeline.process('photo.jpg', output_dir='./results')

# Access tensors
dslr_tensor = results['dslr']
film_tensor = results['film']
```

## Requirements

- Both Stage 1 and Stage 2 models must be trained first
- Checkpoints saved in respective stage directories
- Virtual environment activated

## Files

- `inference.py` - Main inference script
- `INFERENCE_GUIDE.md` - Detailed usage guide

---

**Complete Pipeline** - Smartphone → DSLR → Cinematic Film 🎬

