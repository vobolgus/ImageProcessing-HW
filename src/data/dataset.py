import os
import shutil
import random
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _augment_and_balance_class(class_dir, target_count):
    image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(image_paths)

    if current_count >= target_count:
        print(
            f"    - Class '{os.path.basename(class_dir)}' already has {current_count} images (target: {target_count}). Augmentation not required.")
        return

    num_to_generate = target_count - current_count
    print(
        f"    - Class '{os.path.basename(class_dir)}' has {current_count} images. Generating {num_to_generate} new ones...")

    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

    for i in tqdm(range(num_to_generate)):
        source_image_path = random.choice(image_paths)
        image = Image.open(source_image_path).convert('RGB')
        augmented_image = augmentation_transform(image)

        base_name = os.path.basename(source_image_path)
        name, ext = os.path.splitext(base_name)
        output_filename = f"{name}_aug_{i + 1}{ext}"
        output_path = os.path.join(class_dir, output_filename)
        augmented_image.save(output_path)


def _copy_files_to_split_dirs(splits, source_class_names, processed_base_dir):
    for split_name, (paths, labels) in splits.items():
        print(f"\nProcessing '{split_name}' split...")
        for i, path in tqdm(enumerate(paths)):
            label = labels[i]
            class_name = source_class_names[label]

            target_dir = os.path.join(processed_base_dir, split_name, class_name)
            os.makedirs(target_dir, exist_ok=True)

            shutil.copy(path, target_dir)
        print(f"Copied {len(paths)} original images.")


def _augment_train_set(processed_base_dir, source_class_names, target_aug_count):
    print("\n--- Augmenting Training Set ---")
    train_dir = os.path.join(processed_base_dir, 'train')
    train_class_dirs = [os.path.join(train_dir, name) for name in source_class_names]

    if target_aug_count is None:
        counts = [len(os.listdir(d)) for d in train_class_dirs]
        target_aug_count = max(counts)
        print(f"Target count not provided. Balancing to the largest class size: {target_aug_count} images.")

    for class_dir in train_class_dirs:
        _augment_and_balance_class(class_dir, target_aug_count)


def setup_dataset(
        source_base_dir='data',
        processed_base_dir='data/processed',
        source_class_names=None,
        test_size=0.15,
        val_size=0.15,
        target_aug_count=None,
        random_state=42
):
    if source_class_names is None:
        source_class_names = ['mac-merged', 'laptops-merged']

    if os.path.isdir(processed_base_dir):
        print(f"Directory '{processed_base_dir}' already exists. Skipping dataset creation and augmentation.")
        return

    print(f"Creating new dataset structure in '{processed_base_dir}'...")

    all_image_paths = []
    all_labels = []
    for i, class_name in enumerate(source_class_names):
        class_dir = os.path.join(source_base_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Source directory not found: {class_dir}")

        for filename in tqdm(os.listdir(class_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(class_dir, filename))
                all_labels.append(i)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_image_paths, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=random_state, stratify=y_train_val
    )

    splits = {'train': (X_train, y_train), 'val': (X_val, y_val), 'test': (X_test, y_test)}

    _copy_files_to_split_dirs(splits, source_class_names, processed_base_dir)

    _augment_train_set(processed_base_dir, source_class_names, target_aug_count)

    print("\nDataset preparation complete!")


def get_dataloaders(processed_dir='data/processed', batch_size=32):
    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(f"Processed data directory '{processed_dir}' not found. "
                                f"Run setup_dataset() first.")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(processed_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(processed_dir, 'val'), transform=eval_test_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(processed_dir, 'test'), transform=eval_test_transform)

    print("\nDataset Information:")
    print(f"  Training set: {len(train_dataset)} images in {len(train_dataset.classes)} classes.")
    print(f"  Validation set: {len(val_dataset)} images.")
    print(f"  Test set: {len(test_dataset)} images.")
    print(f"  Classes: {train_dataset.class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\nDataLoaders created with batch size {batch_size}.")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    SOURCE_DIRS = ['mac-merged', 'laptops-merged']
    PROCESSED_DIR = 'data/processed'
    TARGET_AUG_COUNT = 1500
    BATCH_SIZE = 32

    setup_dataset(
        source_base_dir='data',
        processed_base_dir=PROCESSED_DIR,
        source_class_names=SOURCE_DIRS,
        target_aug_count=TARGET_AUG_COUNT
    )

    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            processed_dir=PROCESSED_DIR,
            batch_size=BATCH_SIZE
        )

        print("\n--- Verifying train_loader ---")
        images, labels = next(iter(train_loader))
        print(f"Successfully retrieved one batch.")
        print(f"  Image batch shape: {images.shape}")
        print(f"  Label batch shape: {labels.shape}")

    except (FileNotFoundError, Exception) as e:
        print(f"\nAn error occurred: {e}")

