import os
import shutil
from PIL import Image
from torchvision import transforms
import random
from collections import Counter


def augment_and_balance_dataset(base_dir='data', source_class_dirs=None, target_class_dirs=None, target_cnt=None):
    if source_class_dirs is None:
        source_class_dirs = ['mac-merged', 'laptops-merged']
    if target_class_dirs is None:
        target_class_dirs = ['mac-aug', 'laptops-aug']

    if len(source_class_dirs) != len(target_class_dirs):
        print("Error: The number of source directories must match the number of target directories.")
        return

    for class_name in source_class_dirs:
        if not os.path.isdir(os.path.join(base_dir, class_name)):
            print(f"Error: Source directory '{os.path.join(base_dir, class_name)}' not found.")
            return

    for class_name in target_class_dirs:
        target_path = os.path.join(base_dir, class_name)
        if os.path.isdir(target_path):
            print(f"Warning: Output directory '{target_path}' already exists. It will be removed and recreated.")
            shutil.rmtree(target_path)

    class_counts = {}
    class_paths = {}

    for class_name in source_class_dirs:
        class_dir = os.path.join(base_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts[class_name] = len(images)
        class_paths[class_name] = [os.path.join(class_dir, img) for img in images]

    if not class_counts:
        print(f"Error: No images found in source directories within '{base_dir}'.")
        return

    print("\nOriginal class distribution:")
    for name, count in class_counts.items():
        print(f"  - {name}: {count} images")

    if target_cnt is None:
        majority_class_name = max(class_counts, key=class_counts.get)
        target_count = class_counts[majority_class_name]
        print(f"\nTarget count not provided. Balancing to majority class '{majority_class_name}': {target_count}")
    else:
        target_count = target_cnt
        print(f"\nUsing user-defined target count for all classes: {target_count}")

    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

    for i, class_name in enumerate(source_class_dirs):
        image_count = class_counts[class_name]
        output_class_dir = os.path.join(base_dir, target_class_dirs[i])
        os.makedirs(output_class_dir, exist_ok=True)

        print(f"\nProcessing class: {class_name}")

        if image_count < target_count:
            print(f"  - Copying {image_count} original images to '{output_class_dir}'...")
            for image_path in class_paths[class_name]:
                shutil.copy(image_path, output_class_dir)

            num_to_generate = target_count - image_count
            print(f"  - Augmenting. Need to generate {num_to_generate} new images to reach {target_count}.")

            for j in range(num_to_generate):
                source_image_path = random.choice(class_paths[class_name])
                image = Image.open(source_image_path).convert('RGB')
                augmented_image = augmentation_transform(image)

                base_name = os.path.basename(source_image_path)
                name, ext = os.path.splitext(base_name)
                output_filename = f"{name}_aug_{j + 1}{ext}"
                output_path = os.path.join(output_class_dir, output_filename)
                augmented_image.save(output_path)

                if (j + 1) % 200 == 0:
                    print(f"    ... generated {j + 1}/{num_to_generate} images")

            print(f"  - Finished generating {num_to_generate} images.")

        elif image_count > target_count:
            print(f"  - Down-sampling. Randomly selecting {target_count} of {image_count} images.")
            selected_paths = random.sample(class_paths[class_name], target_count)
            for image_path in selected_paths:
                shutil.copy(image_path, output_class_dir)
            print(f"  - Finished copying {target_count} images.")

        else:
            print(f"  - Size matches target ({target_count}). Copying all original images.")
            for image_path in class_paths[class_name]:
                shutil.copy(image_path, output_class_dir)

    print("\n--- Data processing complete! ---")
    print(f"New dataset is available in the '{target_class_dirs}' directories inside '{base_dir}'.")


if __name__ == '__main__':
    source_dirs = ['mac-merged', 'laptops-merged']
    target_dirs = ['mac-aug', 'laptops-aug']

    augment_and_balance_dataset(
        source_class_dirs=source_dirs,
        target_class_dirs=target_dirs,
        target_cnt=500
    )

