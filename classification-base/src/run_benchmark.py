"""
Benchmark script to run all model/strategy combinations and output results tables.

1st Assignment: ResNet/ViT/Swin/DINO/CLIP with classification head
    - All layers (no freeze)
    - Linear layer only (last/classifier only)
    - 30% of layers (pct70 freeze)
    - LoRA

2nd Assignment: CLIP Contrastive (no classification head)
    - Zero-shot
    - All layers (full fine-tuning)
    - Text encoder only
    - Image encoder only
    - 30% of weights
    - LoRA (all modules)
    - LoRA (attention + mlp only)
"""
import argparse
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from peft import LoraConfig, get_peft_model

import src.train as train_module
from src.data.dataset import (
    setup_dataset_realtime,
    get_STL_dataset,
    AugmentationLevel,
    get_augmentation_transform,
    DatasetBundle,
)
from src.model.freeze_utils import FreezeStrategy, apply_lora, print_trainable_parameters
from src.model.resnet import create_resnet_classifier
from src.model.swin import create_swin_classifier
from src.model.vit import create_vit_classifier
from src.model.dino_vit import create_dinov3_classifier
from src.model.clip import create_clip_classifier
from src.model.clip_contrastive import (
    CLIPFreezeStrategy,
    CLIPContrastiveClassifier,
    create_clip_contrastive_classifier,
    apply_lora_to_clip_contrastive,
)


# Model factories for 1st assignment
MODEL_FACTORIES = {
    'resnet': create_resnet_classifier,
    'vit': create_vit_classifier,
    'swin': create_swin_classifier,
    'dino': create_dinov3_classifier,
    'clip': create_clip_classifier,
}

# Freeze strategies for 1st assignment
FREEZE_STRATEGIES = {
    'all_layers': FreezeStrategy.NO,      # Train all weights
    'linear_layer': FreezeStrategy.LAST,  # Train only classifier head
    '30_layers': FreezeStrategy.PCT70,    # Train 30% of layers (freeze 70%)
    'lora': FreezeStrategy.LORA,          # LoRA adapters
}


@dataclass
class BenchmarkResult:
    model: str
    strategy: str
    test_accuracy: float
    test_f1: float
    best_val_f1: float


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_model(model, dataloader: DataLoader, device: torch.device) -> tuple:
    """Evaluate model and return (accuracy, f1_score)."""
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, f1


def train_and_evaluate(
    model,
    model_name: str,
    bundle: DatasetBundle,
    device: torch.device,
    num_epochs: int,
    lr: float,
    output_dir: str,
) -> BenchmarkResult:
    """Train model and return benchmark result."""
    model = model.to(device)

    # Setup training
    train_module.LR = lr
    train_module.L1_LAMBDA = 1e-6
    train_module.L2_LAMBDA = 1e-5

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) == 0:
        # Zero-shot case - just evaluate
        acc, f1 = evaluate_model(model, bundle.test_loader, device)
        return BenchmarkResult(
            model=model_name.split('-')[0],
            strategy=model_name.split('-')[1] if '-' in model_name else 'zero_shot',
            test_accuracy=acc,
            test_f1=f1,
            best_val_f1=0.0,
        )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)

    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    train_module.train_model(
        model=model,
        model_name=model_name,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        weights_dir=model_dir,
        num_classes=bundle.num_classes,
        class_weights=bundle.class_weights,
        early_stopping=True,
        patience=5,
    )

    # Evaluate on test set
    acc, f1 = evaluate_model(model, bundle.test_loader, device)

    return BenchmarkResult(
        model=model_name.split('-')[0],
        strategy='-'.join(model_name.split('-')[1:]),
        test_accuracy=acc,
        test_f1=f1,
        best_val_f1=f1,  # Approximate
    )


def run_first_assignment(
    bundle: DatasetBundle,
    device: torch.device,
    output_dir: str,
    num_epochs: int = 15,
    lr: float = 1e-4,
) -> pd.DataFrame:
    """Run 1st assignment benchmark: 5 models x 4 strategies."""
    print("\n" + "=" * 70)
    print("1st ASSIGNMENT: Classification Models with Head")
    print("=" * 70)

    results = []

    for model_key, factory in MODEL_FACTORIES.items():
        for strategy_key, strategy in FREEZE_STRATEGIES.items():
            model_name = f"{model_key}-{strategy_key}"
            print(f"\n--- Training: {model_name} ---")

            # Create model
            if strategy == FreezeStrategy.LORA:
                model = factory(num_classes=bundle.num_classes, freeze=None)
                model = apply_lora(model)
                current_lr = 3e-4  # Higher LR for LoRA
            else:
                model = factory(num_classes=bundle.num_classes, freeze=strategy)
                current_lr = lr

            print_trainable_parameters(model)

            result = train_and_evaluate(
                model=model,
                model_name=model_name,
                bundle=bundle,
                device=device,
                num_epochs=num_epochs,
                lr=current_lr,
                output_dir=output_dir,
            )
            results.append(result)

            print(f"Result: Accuracy={result.test_accuracy:.4f}, F1={result.test_f1:.4f}")

    # Create results table
    df = pd.DataFrame([
        {
            'model': r.model,
            'strategy': r.strategy,
            'accuracy': r.test_accuracy,
        }
        for r in results
    ])

    # Pivot to wide format
    pivot_df = df.pivot(index='model', columns='strategy', values='accuracy')
    pivot_df = pivot_df[['all_layers', 'linear_layer', '30_layers', 'lora']]  # Reorder columns

    return pivot_df


def apply_lora_attention_mlp(model: CLIPContrastiveClassifier) -> CLIPContrastiveClassifier:
    """Apply LoRA only to attention (q,k,v) and MLP layers."""
    config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'],
        lora_dropout=0.1,
    )
    model.model = get_peft_model(model.model, config)
    return model


def run_second_assignment(
    bundle: DatasetBundle,
    device: torch.device,
    output_dir: str,
    num_epochs: int = 15,
    lr: float = 3e-5,
) -> pd.DataFrame:
    """Run 2nd assignment benchmark: CLIP Contrastive with 7 strategies."""
    print("\n" + "=" * 70)
    print("2nd ASSIGNMENT: CLIP Contrastive (No Classification Head)")
    print("=" * 70)

    class_labels = bundle.classes
    results = []

    strategies = [
        ('zero_shot', CLIPFreezeStrategy.ZERO_SHOT, lr, False),
        ('all_layers', CLIPFreezeStrategy.FULL, 3e-6, False),
        ('text_encoder', CLIPFreezeStrategy.TEXT_ONLY, 3e-5, False),
        ('image_encoder', CLIPFreezeStrategy.VISION_ONLY, 3e-5, False),
        ('30_weights', CLIPFreezeStrategy.PCT30, 3e-5, False),
        ('lora_all', CLIPFreezeStrategy.LORA, 3e-4, True),  # LoRA all-linear
        ('lora_attn_mlp', CLIPFreezeStrategy.LORA, 3e-4, False),  # LoRA attention+mlp
    ]

    for strategy_name, strategy, current_lr, use_all_linear_lora in strategies:
        model_name = f"clip_contrastive-{strategy_name}"
        print(f"\n--- Training: {model_name} ---")

        # Create model
        model = create_clip_contrastive_classifier(
            labels=class_labels,
            freeze=strategy,
        )

        # Apply LoRA if needed
        if strategy == CLIPFreezeStrategy.LORA:
            if use_all_linear_lora:
                model = apply_lora_to_clip_contrastive(model)
            else:
                model = apply_lora_attention_mlp(model)

        print_trainable_parameters(model)

        result = train_and_evaluate(
            model=model,
            model_name=model_name,
            bundle=bundle,
            device=device,
            num_epochs=num_epochs,
            lr=current_lr,
            output_dir=output_dir,
        )
        results.append((strategy_name, result.test_accuracy))

        print(f"Result: Accuracy={result.test_accuracy:.4f}")

    # Create results DataFrame
    df = pd.DataFrame(results, columns=['strategy', 'accuracy'])
    return df


def print_results_tables(df1: pd.DataFrame, df2: pd.DataFrame):
    """Print formatted results tables."""
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n1st Assignment Results:")
    print("-" * 60)
    print("model".ljust(10) + "all_layers".ljust(12) + "linear_layer".ljust(14) +
          "30_layers".ljust(12) + "lora".ljust(10))
    print("-" * 60)
    for model in ['clip', 'vit', 'swin', 'dino', 'resnet']:
        if model in df1.index:
            row = df1.loc[model]
            print(f"{model.ljust(10)}{row.get('all_layers', 0):.4f}".ljust(22) +
                  f"{row.get('linear_layer', 0):.4f}".ljust(14) +
                  f"{row.get('30_layers', 0):.4f}".ljust(12) +
                  f"{row.get('lora', 0):.4f}")

    print("\n2nd Assignment Results (CLIP Contrastive):")
    print("-" * 40)
    for _, row in df2.iterrows():
        print(f"{row['strategy'].ljust(20)}: {row['accuracy']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run full benchmark for classification assignment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data-root', '-d',
        type=str,
        default='data/LaptopsVsMac',
        help='Path to dataset root directory'
    )
    parser.add_argument(
        '--dataset-type',
        choices=['custom', 'stl10'],
        default='custom',
        help='Type of dataset'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=15,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='models/benchmark',
        help='Output directory'
    )
    parser.add_argument(
        '--assignment',
        choices=['1', '2', 'all'],
        default='all',
        help='Which assignment to run (1, 2, or all)'
    )
    parser.add_argument(
        '--augmentation',
        choices=['light', 'medium', 'heavy'],
        default='medium',
        help='Data augmentation level'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'benchmark-{run_timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Prepare dataset
    print("Preparing dataset...")
    aug_level = AugmentationLevel[args.augmentation.upper()]

    if args.dataset_type == 'stl10':
        ds = get_STL_dataset(args.data_root)
    else:
        ds = datasets.ImageFolder(root=args.data_root)

    aug_transform = get_augmentation_transform(aug_level)
    bundle = setup_dataset_realtime(
        dataset=ds,
        batch_size=args.batch_size,
        augmentation_transform=aug_transform,
    )

    print(f"Classes: {bundle.classes}")
    print(f"Num classes: {bundle.num_classes}")
    print(f"Train size: {len(bundle.train_dataset)}")
    print(f"Val size: {len(bundle.val_dataset)}")
    print(f"Test size: {len(bundle.test_dataset)}")

    device = get_device()
    print(f"Device: {device}")

    df1, df2 = None, None

    # Run assignments
    if args.assignment in ['1', 'all']:
        df1 = run_first_assignment(
            bundle=bundle,
            device=device,
            output_dir=os.path.join(output_dir, 'assignment1'),
            num_epochs=args.epochs,
        )

    if args.assignment in ['2', 'all']:
        df2 = run_second_assignment(
            bundle=bundle,
            device=device,
            output_dir=os.path.join(output_dir, 'assignment2'),
            num_epochs=args.epochs,
        )

    # Print results
    if df1 is not None and df2 is not None:
        print_results_tables(df1, df2)
    elif df1 is not None:
        print("\n1st Assignment Results:")
        print(df1.round(4).to_string())
    elif df2 is not None:
        print("\n2nd Assignment Results:")
        print(df2.round(4).to_string())

    # Save results
    if df1 is not None:
        df1.to_csv(os.path.join(output_dir, 'assignment1_results.csv'))
    if df2 is not None:
        df2.to_csv(os.path.join(output_dir, 'assignment2_results.csv'), index=False)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()