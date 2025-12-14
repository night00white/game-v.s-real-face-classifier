#!/usr/bin/env python3
import sys
import os
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from torchvision import transforms
from classifier_helper import Classifier
import torch


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


SCRIPT_DIR = Path(__file__).parent
BASE_FOLDER = SCRIPT_DIR.parent / 'datasets'
REAL100_FOLDER = BASE_FOLDER / 'real100'
GAME100_FOLDER = BASE_FOLDER / 'game100'
TEMP_DATASET_FOLDER = BASE_FOLDER / 'game_vs_real_vit'
MODEL_FOLDER = SCRIPT_DIR.parent / 'model_vit'

NUMBER_OF_EPOCHS = 25
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_WORKERS = 4
MAX_IMAGES = None

RANDOM_DATA_ROTATIONS_DEG = 10
RANDOM_DATA_RESIZE = (0.8, 1.0)

MODEL_TYPE = 'vit_b_32'
CONTINUE_TRAINING = False

if __name__ == '__main__':
    DEVICE = resolve_device()
    print(f"Device: {DEVICE}")
    print(f"Training binary classifier (ViT-B/32): real=0, game=1")
    
    print(f"\nPreparing dataset from real100 and game100...")
    
    if TEMP_DATASET_FOLDER.exists():
        shutil.rmtree(TEMP_DATASET_FOLDER)
    
    real_dest = TEMP_DATASET_FOLDER / 'real'
    game_dest = TEMP_DATASET_FOLDER / 'game'
    real_dest.mkdir(parents=True, exist_ok=True)
    game_dest.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying real images from {REAL100_FOLDER}...")
    for img in REAL100_FOLDER.glob('*.png'):
        shutil.copy(img, real_dest / img.name)
    
    print(f"Copying game images from {GAME100_FOLDER}...")
    for img in GAME100_FOLDER.glob('*.png'):
        shutil.copy(img, game_dest / img.name)
    
    real_count = len(list(real_dest.glob('*.png')))
    game_count = len(list(game_dest.glob('*.png')))
    print(f"Dataset prepared: {real_count} real, {game_count} game images")
    
    classifier = Classifier.createBinaryClassifier(
        str(MODEL_FOLDER), 
        device=DEVICE,
        model_type=MODEL_TYPE
    )
    
    TRANSFORMS = [
        transforms.RandomRotation(RANDOM_DATA_ROTATIONS_DEG),
        transforms.RandomResizedCrop(224, scale=RANDOM_DATA_RESIZE, ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    ]
    
    print(f"\nTraining on: {TEMP_DATASET_FOLDER}")
    print(f"Epochs: {NUMBER_OF_EPOCHS}, Batch size: {BATCH_SIZE}")
    
    classifier.train(
        NUMBER_OF_EPOCHS,
        str(TEMP_DATASET_FOLDER),
        CONTINUE_TRAINING,
        TRANSFORMS,
        BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_workers=NUM_WORKERS,
        max_images=MAX_IMAGES,
    )
    
    print("\nTraining complete. Exporting TorchScript model...")
    
    classifier_cpu = Classifier.loadFromFolder(str(MODEL_FOLDER), device=torch.device("cpu"))
    classifier_cpu.saveJIT()
    
    print(f"\nModel saved to: {MODEL_FOLDER}")
    print(f"TorchScript model: {MODEL_FOLDER}/model_traced.pt")
    print(f"\nCleaning up temporary dataset folder...")
    shutil.rmtree(TEMP_DATASET_FOLDER)
    print("Done!")
