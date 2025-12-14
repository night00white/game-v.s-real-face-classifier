#!/usr/bin/env python3
import csv
import math
from pathlib import Path
from PIL import Image
import torch
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from classifier_helper import Classifier


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def evaluate_folder(folder: Path, classifier: Classifier):
    rows = []
    summary = {'Game': 0, 'Real': 0, 'error': 0}
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    
    for img_path in files:
        try:
            img = Image.open(img_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            proba = classifier.predict_proba_pil(img)
            prediction = 'Real' if proba['real'] > proba['game'] else 'Game'
            rows.append({
                'folder': folder.name,
                'file': img_path.name,
                'prediction': prediction,
                'real_prob': f"{proba['real']:.4f}",
                'game_prob': f"{proba['game']:.4f}",
                'error': ''
            })
            summary[prediction] += 1
        except Exception as exc:
            rows.append({
                'folder': folder.name,
                'file': img_path.name,
                'prediction': 'error',
                'real_prob': 'n/a',
                'game_prob': 'n/a',
                'error': str(exc)
            })
            summary['error'] += 1
    
    return rows, summary


def create_grid_visualization(folder: Path, classifier: Classifier, output_path: Path):
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    if not images:
        print(f"No images in {folder}")
        return
    
    grid_cols = 5
    max_rows = 6
    max_imgs = grid_cols * max_rows
    images = images[:max_imgs]
    rows = math.ceil(len(images) / grid_cols)
    
    fig = plt.figure(figsize=(grid_cols * 3.2, rows * 3.2))
    fig.suptitle(f'{folder.name} predictions ({len(images)} images)', fontsize=16, fontweight='bold')
    
    for idx, img_path in enumerate(images, 1):
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        proba = classifier.predict_proba_pil(img)
        prediction = 'Real' if proba['real'] > proba['game'] else 'Game'
        
        # Assume test images are game renders
        is_correct = prediction == 'Game'
        color = 'green' if is_correct else 'red'
        status = '✓' if is_correct else '✗'
        
        ax = fig.add_subplot(rows, grid_cols, idx)
        ax.imshow(img)
        title = f"{status} {prediction}\nR:{proba['real']:.2f} G:{proba['game']:.2f}"
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Grid saved: {output_path}")


def main():
    SCRIPT_DIR = Path(__file__).parent
    BASE_DIR = SCRIPT_DIR.parent
    TEST_DATA_DIR = BASE_DIR / 'test_data' / 'Test50'
    MODEL_DIR = BASE_DIR / 'model'
    OUTPUT_DIR = BASE_DIR / 'outputs'
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load model
    device = resolve_device()
    print(f"Device: {device}")
    print(f"Loading model from: {MODEL_DIR}")
    classifier = Classifier.loadFromFolder(str(MODEL_DIR), device)
    
    # Process test1 and test2
    folders = ['test1', 'test2']
    all_results = []
    
    for folder_name in folders:
        folder = TEST_DATA_DIR / folder_name
        if not folder.exists():
            print(f"Skip {folder_name} (not found)")
            continue
        
        print(f"\nProcessing {folder_name}...")
        
        # Evaluate
        rows, summary = evaluate_folder(folder, classifier)
        all_results.extend(rows)
        
        # Save individual CSV
        csv_path = OUTPUT_DIR / f'{folder_name}_predictions.csv'
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['folder', 'file', 'prediction', 'real_prob', 'game_prob', 'error'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV saved: {csv_path}")
        print(f"  Results: {len(rows)} images | Game: {summary['Game']} | Real: {summary['Real']} | Errors: {summary['error']}")
        
        # Create grid visualization
        grid_path = OUTPUT_DIR / f'{folder_name}_grid.png'
        create_grid_visualization(folder, classifier, grid_path)
    
    # Save combined CSV
    if all_results:
        combined_csv = OUTPUT_DIR / 'test50_all_predictions.csv'
        with combined_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['folder', 'file', 'prediction', 'real_prob', 'game_prob', 'error'])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nCombined CSV saved: {combined_csv}")
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
