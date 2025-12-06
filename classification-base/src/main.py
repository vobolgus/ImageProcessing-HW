import argparse
import os
from datetime import datetime
from typing import List, Dict, Callable

# Torch / Lightning bits are used indirectly via train.py
import torch
from torchvision import datasets

import src.train as train_module
from src.data.dataset import setup_dataset_realtime, get_STL_dataset, AugmentationLevel
from src.model.freeze_utils import FreezeStrategy
from src.model.resnet import create_resnet_classifier
from src.model.swin import create_swin_classifier
from src.model.vit import create_vit_classifier
from src.model.dino_vit import create_dinov3_classifier
from src.test import run_tests


# All available model factories
MODEL_FACTORIES: Dict[str, Callable] = {
    'resnet': create_resnet_classifier,
    'swin': create_swin_classifier,
    'vit': create_vit_classifier,
    'dinov3': create_dinov3_classifier,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train image classification models with various backbones.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model selection
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=list(MODEL_FACTORIES.keys()) + ['all'],
        default=['all'],
        help='Models to train. Use "all" to train all available models.'
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
        default=1e-4,
        help='Learning rate'
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

    # Freeze strategy
    parser.add_argument(
        '--freeze',
        choices=['no', 'last', 'pct70'],
        default='pct70',
        help='Freeze strategy for transfer learning'
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


def get_freeze_strategy(name: str) -> FreezeStrategy:
    mapping = {
        'no': FreezeStrategy.NO,
        'last': FreezeStrategy.LAST,
        'pct70': FreezeStrategy.PCT70,
    }
    return mapping[name.lower()]


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

    # Get augmentation transform based on level
    from src.data.dataset import get_augmentation_transform
    aug_transform = get_augmentation_transform(augmentation_level)

    bundle = setup_dataset_realtime(
        dataset=ds,
        batch_size=batch_size,
        augmentation_transform=aug_transform,
    )
    return bundle


def main():
    args = parse_args()

    # Determine which models to train
    if 'all' in args.models:
        models_to_train = MODEL_FACTORIES
    else:
        models_to_train = {name: MODEL_FACTORIES[name] for name in args.models}

    # Parse freeze strategy and augmentation level
    freeze_strategy = get_freeze_strategy(args.freeze)
    aug_level = get_augmentation_level(args.augmentation)

    # Create run-scoped output directory
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'run-{run_timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"  Models:          {list(models_to_train.keys())}")
    print(f"  Dataset:         {args.data_root} ({args.dataset_type})")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Freeze strategy: {args.freeze}")
    print(f"  Augmentation:    {args.augmentation}")
    print(f"  Early stopping:  {args.early_stopping} (patience={args.patience})")
    print(f"  Output dir:      {run_dir}")
    print("=" * 60)

    # Prepare dataset once
    bundle = prepare_dataset(args.dataset_type, args.data_root, args.batch_size, aug_level)
    effective_num_classes = bundle.num_classes

    device = train_module.get_device()
    print(f"Selected device: {device}")

    # Propagate hyperparameters to training module
    train_module.LR = args.lr
    train_module.L1_LAMBDA = args.l1_lambda
    train_module.L2_LAMBDA = args.l2_lambda

    trained_model_prefixes: List[str] = []

    for name, factory in models_to_train.items():
        model = factory(num_classes=effective_num_classes, freeze=freeze_strategy)
        model = model.to(device)

        freeze_str = freeze_strategy.name.lower()
        model_name = f"{name.lower()}-{freeze_str}"
        trained_model_prefixes.append(model_name)

        # Variant-specific directory
        model_variant_dir = os.path.join(run_dir, model_name)
        os.makedirs(model_variant_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Training {model_name} for {args.epochs} epochs")
        print(f"{'=' * 60}")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_lambda)

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
            num_classes=effective_num_classes,
            class_weights=bundle.class_weights,
            early_stopping=args.early_stopping,
            patience=args.patience,
        )

    # After training all models, run tests
    print(f"\n{'=' * 60}")
    print("Running tests for trained models")
    print(f"{'=' * 60}")
    run_tests(
        weights_dir=run_dir,
        results_csv_path=os.path.join(run_dir, 'test_results.csv'),
        include_prefixes=trained_model_prefixes,
        bundle=bundle,
    )


if __name__ == '__main__':
    main()
