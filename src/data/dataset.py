import os
import shutil
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _augment_single_image(args):
    """
    Worker function to augment a single image. It's designed to be called
    by a ProcessPoolExecutor.
    """
    source_image_path, class_dir, augmentation_transform, i = args
    try:
        image = Image.open(source_image_path).convert('RGB')
        augmented_image = augmentation_transform(image)

        # Use a completely new, unique name to avoid any conflicts
        output_filename = f"aug_set_{random.randint(10000, 99999)}_{i}.jpg"
        output_path = os.path.join(class_dir, output_filename)

        augmented_image.save(output_path, "JPEG")  # Save as JPEG for consistency
        return output_path
    except Exception as e:
        # Return error message instead of path
        return f"Error processing {source_image_path}: {e}"


def _augment_and_balance_class(class_dir, target_count):
    """
    Replaces the original images in a class directory with a new, fully augmented
    set of images of a specific target count, processed in parallel.
    """
    print(f"\nProcessing class: '{os.path.basename(class_dir)}'")
    original_image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if
                            f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(original_image_paths)
    print(f"  - Initial image count: {current_count}")

    if current_count == 0:
        print("    - Warning: No images found in directory to augment. Skipping.")
        return

    if target_count == 0:
        print("   - Warning: target_count is 0. All original images will be deleted.")
        for f_path in tqdm(original_image_paths, desc="Deleting all images"):
            os.remove(f_path)
        return

    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

    # --- STAGE 1: Generate a new, fully augmented dataset in parallel ---
    print(f"  - Stage 1: Generating {target_count} new augmented images...")
    repeats = target_count // current_count
    remainder = target_count % current_count
    source_list = original_image_paths * repeats + original_image_paths[:remainder]
    random.shuffle(source_list)

    new_image_paths = []
    tasks = [(path, class_dir, augmentation_transform, i) for i, path in enumerate(source_list)]

    num_workers = multiprocessing.cpu_count()
    print(f"    - Using {num_workers} workers for parallel augmentation...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_augment_single_image, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Generating new set"):
            result = future.result()
            if isinstance(result, str) and not result.startswith("Error"):
                new_image_paths.append(result)
            elif isinstance(result, str):
                print(f"\n{result}") # Print augmentation errors

    # --- STAGE 2: Delete all original images ---
    print(f"  - Stage 2: Deleting {len(original_image_paths)} original images...")
    for f_path in tqdm(original_image_paths, desc="Deleting originals"):
        try:
            if f_path not in new_image_paths:
                os.remove(f_path)
        except OSError as e:
            print(f"\nWarning: Could not delete file {f_path}. Error: {e}")

    # Final verification
    final_image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"  - Final image count: {len(final_image_paths)}")


def _copy_files_to_split_dirs(splits, source_class_names, processed_base_dir):
    for split_name, (paths, labels) in splits.items():
        print(f"\nProcessing '{split_name}' split...")
        for i, path in tqdm(enumerate(paths), total=len(paths)):
            label = labels[i]
            class_name = source_class_names[label]

            target_dir = os.path.join(processed_base_dir, split_name, class_name)
            os.makedirs(target_dir, exist_ok=True)

            shutil.copy(path, target_dir)
        print(f"Copied {len(paths)} original images.")


def _augment_train_set(processed_base_dir, source_class_names, target_aug_count):
    print("\n--- Augmenting and Balancing Training Set ---")
    train_dir = os.path.join(processed_base_dir, 'train')
    train_class_dirs = [os.path.join(train_dir, name) for name in source_class_names]

    if target_aug_count is None:
        counts = [len(os.listdir(d)) for d in train_class_dirs if os.path.isdir(d)]
        if counts:
            target_aug_count = max(counts)
            print(f"Target count not provided. Balancing to the largest class size: {target_aug_count} images.")
        else:
            print("Warning: No class directories found to determine target count.")
            return

    for class_dir in train_class_dirs:
        if os.path.isdir(class_dir):
            _augment_and_balance_class(class_dir, target_aug_count)
        else:
            print(f"Warning: Class directory not found, skipping: {class_dir}")


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
        # Random augmentations are still useful during training
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"\nDataLoaders created with batch size {batch_size}.")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    SOURCE_DIRS = ['mac-merged', 'laptops-merged']
    PROCESSED_DIR = 'data/processed123'
    TARGET_AUG_COUNT = 1024
    BATCH_SIZE = 32

    # Run setup
    setup_dataset(
        source_base_dir='data',
        processed_base_dir=PROCESSED_DIR,
        source_class_names=SOURCE_DIRS,
        target_aug_count=TARGET_AUG_COUNT
    )

    try:
        # Verify dataloaders
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

