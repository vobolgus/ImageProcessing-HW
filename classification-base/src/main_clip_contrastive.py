"""Training script for CLIP Contrastive model (without classification head).

This script trains CLIP using the contrastive approach where classification
is performed by comparing image embeddings with text embeddings of class prompts.

Implements 6 training strategies:
1) Zero Shot - No training, use pretrained model
2) PCT30 - Train 30% of last layers (both encoders)
3) Full - Train all weights (ImageEncoder + TextEncoder)
4) Vision Only - Train only ImageEncoder
5) Text Only - Train only TextEncoder
6) LoRA - Use LoRA adapters
"""
import argparse
import os
from datetime import datetime
from typing import List

import torch
from torchvision import datasets

import src.train as train_module
from src.data.dataset import setup_dataset_realtime, get_STL_dataset, AugmentationLevel, get_augmentation_transform
from src.model.clip_contrastive import (
    CLIPFreezeStrategy,
    create_clip_contrastive_classifier,
    apply_lora_to_clip_contrastive,
    print_trainable_parameters,
)
from src.test import run_tests


# Available CLIP contrastive freeze strategies
CLIP_STRATEGIES = {
    'zero_shot': CLIPFreezeStrategy.ZERO_SHOT,
    'pct30': CLIPFreezeStrategy.PCT30,
    'full': CLIPFreezeStrategy.FULL,
    'vision_only': CLIPFreezeStrategy.VISION_ONLY,
    'text_only': CLIPFreezeStrategy.TEXT_ONLY,
    'lora': CLIPFreezeStrategy.LORA,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CLIP contrastive model with various freeze strategies.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Strategy selection
    parser.add_argument(
        '--strategies', '-s',
        nargs='+',
        choices=list(CLIP_STRATEGIES.keys()) + ['all'],
        default=['all'],
        help='CLIP training strategies to run. Use "all" to run all strategies.'
    )

    # Data settings
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
        help='Type of dataset: "custom" for ImageFolder, "stl10" for STL10'
    )

    # Training hyperparameters
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
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=3e-5,
        help='Learning rate (3e-5 recommended for CLIP fine-tuning)'
    )
    parser.add_argument(
        '--lr-lora',
        type=float,
        default=3e-4,
        help='Learning rate for LoRA training (typically higher)'
    )
    parser.add_argument(
        '--l1-lambda',
        type=float,
        default=1e-6,
        help='L1 regularization coefficient'
    )
    parser.add_argument(
        '--l2-lambda',
        type=float,
        default=1e-5,
        help='L2 regularization (weight decay) coefficient'
    )

    # Augmentation
    parser.add_argument(
        '--augmentation',
        choices=['light', 'medium', 'heavy'],
        default='medium',
        help='Data augmentation intensity level'
    )

    # Early stopping
    parser.add_argument(
        '--early-stopping',
        action='store_true',
        default=True,
        help='Enable early stopping based on validation F1'
    )
    parser.add_argument(
        '--no-early-stopping',
        action='store_false',
        dest='early_stopping',
        help='Disable early stopping'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience (epochs without improvement)'
    )

    # Output
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='models',
        help='Base output directory for run results'
    )

    return parser.parse_args()


def get_augmentation_level(name: str) -> AugmentationLevel:
    mapping = {
        'light': AugmentationLevel.LIGHT,
        'medium': AugmentationLevel.MEDIUM,
        'heavy': AugmentationLevel.HEAVY,
    }
    return mapping[name.lower()]


def prepare_dataset(dataset_type: str, data_root: str, batch_size: int, augmentation_level: AugmentationLevel):
    """Prepare dataset with the specified augmentation level."""
    if dataset_type.lower() == 'stl10':
        ds = get_STL_dataset(data_root)
    else:
        ds = datasets.ImageFolder(root=data_root)

    aug_transform = get_augmentation_transform(augmentation_level)

    bundle = setup_dataset_realtime(
        dataset=ds,
        batch_size=batch_size,
        augmentation_transform=aug_transform,
    )
    return bundle


def main():
    args = parse_args()

    # Determine which strategies to run
    if 'all' in args.strategies:
        strategies_to_run = CLIP_STRATEGIES
    else:
        strategies_to_run = {name: CLIP_STRATEGIES[name] for name in args.strategies}

    aug_level = get_augmentation_level(args.augmentation)

    # Create run-scoped output directory
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'clip-contrastive-{run_timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print("CLIP Contrastive Training Configuration")
    print("=" * 60)
    print(f"  Strategies:      {list(strategies_to_run.keys())}")
    print(f"  Dataset:         {args.data_root} ({args.dataset_type})")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  LR (LoRA):       {args.lr_lora}")
    print(f"  Augmentation:    {args.augmentation}")
    print(f"  Early stopping:  {args.early_stopping} (patience={args.patience})")
    print(f"  Output dir:      {run_dir}")
    print("=" * 60)

    # Prepare dataset once
    bundle = prepare_dataset(args.dataset_type, args.data_root, args.batch_size, aug_level)

    # Get class labels from the dataset
    class_labels = bundle.classes
    print(f"\nClass labels: {class_labels}")
    print(f"Number of classes: {len(class_labels)}")

    device = train_module.get_device()
    print(f"Selected device: {device}")

    # Propagate hyperparameters to training module
    train_module.L1_LAMBDA = args.l1_lambda
    train_module.L2_LAMBDA = args.l2_lambda

    trained_model_prefixes: List[str] = []

    for strategy_name, strategy in strategies_to_run.items():
        print(f"\n{'=' * 60}")
        print(f"Strategy: {strategy_name}")
        print(f"{'=' * 60}")

        # Create model with appropriate freeze strategy
        model = create_clip_contrastive_classifier(
            labels=class_labels,
            freeze=strategy,
        )

        # Apply LoRA if needed
        if strategy == CLIPFreezeStrategy.LORA:
            model = apply_lora_to_clip_contrastive(model)
            lr = args.lr_lora
        else:
            lr = args.lr

        model = model.to(device)
        print_trainable_parameters(model)

        model_name = f"clip-contrastive-{strategy_name}"
        trained_model_prefixes.append(model_name)

        # Create output directory for this strategy
        model_variant_dir = os.path.join(run_dir, model_name)
        os.makedirs(model_variant_dir, exist_ok=True)

        # Handle zero-shot (no training)
        if strategy == CLIPFreezeStrategy.ZERO_SHOT:
            print("Zero-shot mode: Skipping training, evaluating directly...")
            # Just save the model state
            save_path = os.path.join(model_variant_dir, f"{model_name}-zero_shot.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            continue

        print(f"\nTraining {model_name} for {args.epochs} epochs with lr={lr}")

        # Set learning rate for this strategy
        train_module.LR = lr

        criterion = torch.nn.CrossEntropyLoss()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=args.l2_lambda)

        train_module.train_model(
            model=model,
            model_name=model_name,
            train_loader=bundle.train_loader,
            val_loader=bundle.val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=args.epochs,
            weights_dir=model_variant_dir,
            num_classes=len(class_labels),
            class_weights=bundle.class_weights,
            early_stopping=args.early_stopping,
            patience=args.patience,
        )

    # After training all strategies, run tests
    print(f"\n{'=' * 60}")
    print("Running tests for trained models")
    print(f"{'=' * 60}")

    # Custom test for CLIP contrastive models
    from src.test_clip_contrastive import run_clip_contrastive_tests
    run_clip_contrastive_tests(
        weights_dir=run_dir,
        results_csv_path=os.path.join(run_dir, 'test_results.csv'),
        bundle=bundle,
        class_labels=class_labels,
    )


if __name__ == '__main__':
    main()