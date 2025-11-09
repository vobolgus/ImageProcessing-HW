import os
from typing import Optional

import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import torch
from torch import nn
from torchvision import transforms as T
import albumentations  # TTA: Импортируем albumentations

from data import visualize
from lightning_module import CovidSegmenter


def predict_batch(model: nn.Module, image_batch_t: torch.Tensor) -> torch.Tensor:
    """
    TTA: Новая функция, которая просто прогоняет батч тензоров через модель.
    """
    model.eval()
    with torch.no_grad():
        output = model(image_batch_t)
        # Применяем Softmax ЗДЕСЬ, чтобы усреднять вероятности
        output = nn.Softmax(dim=1)(output)
    return output


def run_test_predictions(checkpoint_callback, datamodule, device, target_size, miou_val: Optional[float]):
    print("\nStarting test predictions...")

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path or not os.path.exists(best_model_path):
        print("No best model checkpoint found. Skipping test predictions.")
        return

    print(f"Loading best model from: {best_model_path}")
    best_model = CovidSegmenter.load_from_checkpoint(best_model_path)
    best_model.to(device)
    best_model.eval()

    datamodule.setup('fit')
    test_images = datamodule.test_images
    val_augs = datamodule.val_augs  # Это наш ресайз

    # TTA: Определяем трансформацию (нормализацию)
    mean = [0.485]
    std = [0.229]
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    # TTA: Определяем аугментацию для флипа
    hflip_aug = albumentations.HorizontalFlip(p=1.0)

    print(f"Running predictions with TTA (Horizontal Flip) on {len(test_images)} images...")

    output = np.zeros((len(test_images), target_size, target_size, 4))

    for i in range(len(test_images)):
        img_np = test_images[i]  # Оригинальное numpy изображение

        # --- 1. Оригинальное изображение ---
        img_orig_aug = val_augs(image=img_np)['image']  # Применяем ресайз
        img_orig_t = transform(img_orig_aug).to(device)  # Конвертируем в тензор

        # --- 2. Отраженное изображение ---
        img_flipped_np = hflip_aug(image=img_np)['image']  # Отражаем numpy
        img_flipped_aug = val_augs(image=img_flipped_np)['image']  # Применяем ресайз
        img_flipped_t = transform(img_flipped_aug).to(device)  # Конвертируем в тензор

        # --- 3. Пакетное предсказание ---
        # Стэкаем оба изображения в один батч [2, 1, H, W]
        batch_t = torch.stack([img_orig_t, img_flipped_t])

        # Получаем предсказания [2, 4, H, W]
        preds_t = predict_batch(best_model, batch_t)

        # --- 4. Усреднение ---
        pred_orig_t = preds_t[0]  # [4, H, W]

        # Берем отраженное предсказание и отражаем его обратно по оси W (dim=2)
        pred_flipped_restored_t = torch.flip(preds_t[1], dims=[2])  # [4, H, W]

        # Усредняем
        avg_pred_t = (pred_orig_t + pred_flipped_restored_t) / 2.0

        # --- 5. Сохранение результата ---
        # Конвертируем в numpy [H, W, 4] для сохранения
        avg_pred_np = avg_pred_t.permute(1, 2, 0).cpu().numpy()
        output[i] = avg_pred_np

    print(f"Output prediction shape (after TTA): {output.shape}")
    test_masks_prediction = output > 0.5

    # Визуализируем TTA-результаты
    # Нам нужен image_batch для visualize, создадим его из оригинальных аугментированных картинок
    image_batch_orig = np.stack([val_augs(image=img)['image'] for img in test_images], axis=0)
    visualize(image_batch_orig, test_masks_prediction, num_samples=len(test_images))

    print("Resizing test predictions to original size...")
    # TTA: Убедимся, что обрезаем 4-й класс (если он есть), но код ноутбука не обрезал...
    # Судя по вашему коду, у вас 4 класса, и вы обрезали 2. Оставим эту логику.
    # test_masks_prediction_original_size = scipy.ndimage.zoom(test_masks_prediction, (1, 2, 2, 1), order=0)

    # Код из ноутбука (и ваш) обрезает последние 2 класса. Это странно, но оставим как есть.
    test_masks_prediction_original_size = scipy.ndimage.zoom(test_masks_prediction[..., :-2], (1, 2, 2, 1), order=0)
    print(f"Resized predictions shape: {test_masks_prediction_original_size.shape}")

    print("Creating submission file (sub.csv)...")
    frame = pd.DataFrame(
        data=np.stack(
            (np.arange(len(test_masks_prediction_original_size.ravel())),
             test_masks_prediction_original_size.ravel().astype(int)),
            axis=1
        ),
        columns=('Id', 'Predicted')
    )
    frame.to_csv('sub.csv', index=False)

    print("Submission file created.")

    if miou_val:
        # TTA: Добавляем TTA_ в имя файла, чтобы отличать логи
        log_filename = f"submission_miou_{miou_val:.4f}_TTA.csv"
        frame.to_csv(log_filename, index=False)
        print(f"Submission file also saved to {log_filename}")