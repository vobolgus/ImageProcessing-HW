"""Testing script for CLIP Contrastive models."""
import os
import re
from typing import Dict, Tuple, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.clip_contrastive import (
    create_clip_contrastive_classifier,
    CLIPFreezeStrategy,
)
from src.data.dataset import DatasetBundle


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA.")
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS.")
        return torch.device("mps")
    print("Using CPU.")
    return torch.device("cpu")


def find_clip_contrastive_models(weights_dir: str) -> Dict[str, Dict[str, any]]:
    """Find all CLIP contrastive model checkpoints."""
    best_models = {}

    # Pattern for trained models: clip-contrastive-{strategy}-DATETIME-f1_SCORE.pth
    trained_pattern = re.compile(r"^(clip-contrastive-.*?)-\d{8}_\d{6}-f1_([\d.]+)\.pth$")
    # Pattern for zero-shot: clip-contrastive-zero_shot-zero_shot.pth
    zero_shot_pattern = re.compile(r"^(clip-contrastive-zero_shot)-zero_shot\.pth$")

    print(f"Searching for CLIP contrastive models in {weights_dir} (recursively)...")

    for root, _, files in os.walk(weights_dir):
        for filename in files:
            if not filename.endswith(".pth"):
                continue

            full_path = os.path.join(root, filename)

            # Check for zero-shot model
            zero_match = zero_shot_pattern.match(filename)
            if zero_match:
                model_name = zero_match.group(1)
                if model_name not in best_models:
                    best_models[model_name] = {
                        'path': full_path,
                        'best_val_f1': 0.0,  # No validation for zero-shot
                        'is_zero_shot': True,
                    }
                continue

            # Check for trained models
            trained_match = trained_pattern.match(filename)
            if trained_match:
                model_name, f1_str = trained_match.groups()
                f1_score_val = float(f1_str)

                if model_name not in best_models or f1_score_val > best_models[model_name]['best_val_f1']:
                    best_models[model_name] = {
                        'path': full_path,
                        'best_val_f1': f1_score_val,
                        'is_zero_shot': False,
                    }

    print(f"Found {len(best_models)} CLIP contrastive models.")
    for name, data in best_models.items():
        if data.get('is_zero_shot'):
            print(f"  - Model: {name} (zero-shot)")
        else:
            print(f"  - Model: {name}, Best Val F1: {data['best_val_f1']:.4f}")

    return best_models


@torch.no_grad()
def test_clip_contrastive_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Test a CLIP contrastive model."""
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Testing"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return test_acc, test_f1


def run_clip_contrastive_tests(
    weights_dir: str,
    results_csv_path: str,
    bundle: DatasetBundle,
    class_labels: List[str],
):
    """Run tests for all CLIP contrastive models."""
    device = get_device()

    best_models_info = find_clip_contrastive_models(weights_dir)
    if not best_models_info:
        print("No CLIP contrastive models found.")
        return None

    test_loader = bundle.test_loader
    results = []

    for model_name, info in best_models_info.items():
        print(f"\n--- Testing model: {model_name} ---")
        try:
            # Create model with labels
            model = create_clip_contrastive_classifier(
                labels=class_labels,
                freeze=None,  # Don't apply any freeze, we're loading weights
            )

            # Load weights
            state_dict = torch.load(info['path'], map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded from: {info['path']}")

            # Update label embeddings after loading (in case they weren't saved properly)
            model._update_label_embeddings()

            test_acc, test_f1 = test_clip_contrastive_model(model, test_loader, device)
            print(f"Test results: Accuracy = {test_acc:.4f}, F1-score = {test_f1:.4f}")

            results.append({
                'model_name': model_name,
                'best_val_f1': info['best_val_f1'],
                'test_accuracy': test_acc,
                'test_f1_score': test_f1,
                'weights_path': info['path']
            })
        except Exception as e:
            print(f"Failed to test model {model_name}. Error: {e}")
            import traceback
            traceback.print_exc()

    if results:
        results_df = pd.DataFrame(results)

        print("\n--- CLIP Contrastive Results ---")
        pd.set_option('display.precision', 4)
        pd.set_option('display.max_colwidth', None)
        print(results_df.to_string(index=False))

        results_df.to_csv(results_csv_path, index=False)
        print(f"\nResults saved to: {results_csv_path}")
        return results_df
    else:
        print("\nNo models were tested.")
        return None


if __name__ == '__main__':
    # Example standalone usage
    from torchvision import datasets
    from src.data.dataset import setup_dataset_realtime

    weights_dir = 'models'
    data_root = 'data/LaptopsVsMac'

    ds = datasets.ImageFolder(root=data_root)
    bundle = setup_dataset_realtime(dataset=ds, batch_size=32)

    run_clip_contrastive_tests(
        weights_dir=weights_dir,
        results_csv_path='clip_contrastive_results.csv',
        bundle=bundle,
        class_labels=bundle.classes,
    )