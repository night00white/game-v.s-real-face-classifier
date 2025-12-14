# Game vs Real Face Classifier - Complete Package

This folder contains everything needed to train and test a binary classifier that distinguishes between real human faces and game/CG-rendered faces.

## Folder Structure

```
game_real_classifier_complete/
├── datasets/           # Training datasets
│   ├── real100/       # 100 real human face images (AdobeStock)
│   └── game100/       # 100 CG/game-rendered face images
├── test_data/         # Test datasets
│   └── Test50/
│       ├── test1/     # 31 test images (CG renders)
│       └── test2/     # 25 test images (mix)
├── model/             # Trained classifier weights
│   ├── model.pth      # PyTorch model weights
│   ├── model_traced.pt # TorchScript model (production-ready)
│   └── params.json    # Model metadata
├── scripts/           # All training/testing scripts
│   ├── classifier_helper.py              # Core classifier class
│   ├── train_binary_game_detector.py     # Training script
│   ├── evaluate_test50.py               # Test50 evaluation + grids
│   └── evaluate_real_dataset.py         # Flexible batch evaluation
└── outputs/           # Generated results (CSVs and grid images)
```

## Quick Start

### Requirements
- Python 3.8+
- PyTorch with torchvision
- PIL/Pillow
- matplotlib

Install dependencies:
```bash
pip install torch torchvision pillow matplotlib
```

### 1. Train the Classifier

Train on real100 + game100 datasets (200 images total):

```bash
cd game_real_classifier_complete
python scripts/train_binary_game_detector.py
```

**Training Configuration:**
- Epochs: 25
- Batch size: 128
- Model: ResNet18 (pretrained on ImageNet)
- Device: Auto-detect (CUDA → MPS → CPU)
- Label mapping: `real=0`, `game=1` (enforced)

The trained model is saved to `model/` folder.

### 2. Test on Test50 Dataset

Run complete evaluation on test1 and test2 folders:

```bash
python scripts/evaluate_test50.py
```

This generates:
- `outputs/test1_predictions.csv` - Predictions for test1
- `outputs/test2_predictions.csv` - Predictions for test2
- `outputs/test50_all_predictions.csv` - Combined results
- `outputs/test1_grid.png` - Visualization grid for test1
- `outputs/test2_grid.png` - Visualization grid for test2

### 3. Single Image Inference

Test a single image:

```bash
python scripts/infer_game_vs_real.py --image path/to/image.jpg --model-folder model/
```

Output example:
```
Prediction probabilities:
  Real: 0.9430
  Game: 0.0570

Prediction: Real (confidence: 0.9430)
```

### 4. Batch Evaluation with CSV

Evaluate a folder of images and export to CSV:

```bash
python scripts/evaluate_real_dataset.py \
  --images-folder path/to/images \
  --output-csv results.csv
```

Add `--only-game` flag to filter only images predicted as game faces.

## Model Performance

**Training Results (real100 + game100):**
- Final accuracy: 95.0% (epoch 24/24)
- Loss: 0.1777
- Training time: ~15 minutes (on Apple M-series GPU)

**Test Results (real100 + game100 validation):**
- Real100 accuracy: 99/100 (99.0%)
- Game100 accuracy: 92/100 (92.0%)
- Overall: 191/200 (95.5%)

**Test50 Results:**
- test1: 30/31 predicted as Game (96.8%)
- test2: 18/25 predicted as Game (72.0%)

## Technical Details

### Label Mapping
The classifier enforces a fixed label mapping:
- `real = 0` (real human faces)
- `game = 1` (CG/game-rendered faces)

This is independent of filesystem folder ordering and ensures consistent predictions.

### Model Architecture
- Backbone: ResNet18 (pretrained on ImageNet)
- Final layer: Custom binary classification head
- Input size: 224×224 RGB images
- Augmentation: Random rotation, crop, flip, color jitter

### Device Support
Auto-detects best available device:
1. CUDA (NVIDIA GPU)
2. MPS (Apple Silicon GPU)
3. CPU (fallback)

### Export Formats
- `model.pth` - Standard PyTorch checkpoint
- `model_traced.pt` - TorchScript (for production deployment)

## File Descriptions

**classifier_helper.py** - Core classifier class with training and inference methods

**train_binary_game_detector.py** - Trains ResNet18 on real100/game100 with data augmentation

**evaluate_test50.py** - Batch evaluation with CSV output and grid visualizations

**evaluate_real_dataset.py** - Flexible batch evaluation for any image folder

## Grid Visualizations

Grid images show:
- ✓ Green = Correct prediction (game image predicted as Game)
- ✗ Red = Incorrect prediction (game image predicted as Real)
- Confidence scores: R:0.xx (Real probability), G:0.xx (Game probability)

## Re-training

To retrain from scratch:

1. Delete existing model:
```bash
rm -rf model/*
```

2. Run training script:
```bash
python scripts/train_binary_game_detector.py
```

Training hyperparameters can be adjusted in `train_binary_game_detector.py`:
- `NUMBER_OF_EPOCHS` (default: 25)
- `BATCH_SIZE` (default: 128)
- `LEARNING_RATE` (default: 0.001)

---

**Created:** December 2025  
**Model Version:** 1.0  
**Framework:** PyTorch 2.x
