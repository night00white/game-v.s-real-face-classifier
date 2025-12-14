import argparse
import csv
import os
from pathlib import Path

import torch
from PIL import Image

from classifier_helper import Classifier


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

MODEL_FOLDER = "/Users/wyl/Downloads/08_optimizers_experiments/02_models/trained_models/art_quick_B"
DATASET_FOLDER = "/Users/wyl/Downloads/08_optimizers_experiments/01_datasets/raw_data/aahq_dataset/raw/real"
OUTPUT_CSV = "/Users/wyl/Downloads/08_optimizers_experiments/04_outputs/evaluation_results/real_dataset_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained classifier on a folder of images.")
    parser.add_argument("--dataset", default=DATASET_FOLDER, help="Folder containing images to classify")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Location to write the CSV results")
    parser.add_argument(
        "--model-folder",
        default=MODEL_FOLDER,
        help="Folder that contains model_weights.pth and model_info.json",
    )
    parser.add_argument(
        "--only-game",
        action="store_true",
        help="Only record results predicted as the 'game' class",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = resolve_device()
    print(f"Using device: {device}")

    classifier = Classifier.loadFromFolder(args.model_folder, device=device)
    classifier.model.eval()

    class_names = classifier.class_names
    if not class_names:
        print("No class names found in the classifier metadata; aborting.")
        return

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset folder not found: {dataset_path}")
        return

    image_paths = sorted(
        [
            path
            for path in dataset_path.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )

    if not image_paths:
        print("No images found to evaluate.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    game_predictions = 0

    for idx, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        logits, probabilities, predicted_idx = classifier.classify(image)
        predicted_class = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        if args.only_game and predicted_class.lower() != "game":
            continue

        results.append(
            {
                "image": str(image_path),
                "predicted_class": predicted_class,
                "confidence_percent": confidence,
            }
        )

        print(
            f"[{idx}/{len(image_paths)}] {image_path.name} -> {predicted_class}"
            f" ({confidence:.2f}% confidence)"
        )

        if predicted_class.lower() == "game":
            game_predictions += 1

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image", "predicted_class", "confidence_percent"])
        writer.writeheader()
        writer.writerows(results)

    print("\nEvaluation complete.")
    if args.only_game:
        print(f"Predicted 'game' for {game_predictions} images.")
    else:
        real_predictions = len(image_paths) - game_predictions
        print(f"Predicted 'real' for {real_predictions} out of {len(image_paths)} images.")
    print(f"Detailed predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
