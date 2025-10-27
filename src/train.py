import os
import warnings
from datetime import datetime
from typing import Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import setup_dataset, get_dataloaders
from src.model.dino_vit import create_dino_vit_classifier
from src.model.dino_convnext import create_dino_swin_classifier

warnings.filterwarnings("ignore", message=".*Redirects are currently not supported in Windows or MacOs.*")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA.")
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS.")
        return torch.device("mps")
    print("Using CPU.")
    return torch.device("cpu")


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: Optimizer,
                    device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss: float = 0.0
    correct_predictions: torch.Tensor = torch.tensor(0)
    total_samples: int = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if L1_LAMBDA > 0:
            l1_loss = sum(param.abs().sum() for param in model.parameters())
            loss += L1_LAMBDA * l1_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

    epoch_loss: float = running_loss / total_samples
    epoch_acc: float = correct_predictions / total_samples

    if device.type == "cuda":
        torch.cuda.synchronize()

    return epoch_loss, epoch_acc


def validate_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[
    float, float, float]:
    model.eval()
    running_loss: float = 0.0
    correct_predictions: torch.Tensor = torch.tensor(0)
    total_samples: int = 0

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss: float = running_loss / total_samples
    epoch_acc: float = correct_predictions / total_samples
    epoch_f1: float = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


def save_and_plot_history(history: List[dict], model_name: str, run_timestamp: str, weights_dir: str):
    history_df = pd.DataFrame(history)
    csv_filename = f"{model_name}-{run_timestamp}-metrics.csv"
    csv_path = os.path.join(weights_dir, csv_filename)
    history_df.to_csv(csv_path, index=False)
    print(f"Metrics history saved to {csv_path}")

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} - Loss vs. Epoch')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_df['epoch'], history_df['train_acc'], label='Train Accuracy')
    plt.plot(history_df['epoch'], history_df['val_acc'], label='Validation Accuracy')
    plt.plot(history_df['epoch'], history_df['val_f1'], label='Validation F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title(f'{model_name} - Metrics vs. Epoch')
    plt.grid(True)

    plt.suptitle(f'Training Metrics for {model_name} ({run_timestamp})')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_filename = f"{model_name}-{run_timestamp}-metrics.png"
    plot_path = os.path.join(weights_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Metrics plot saved to {plot_path}")


def train_model(
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        num_epochs: int,
        weights_dir: str
) -> None:
    best_val_f1: float = 0.0
    os.makedirs(weights_dir, exist_ok=True)
    history = []
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}, Validation F1: {val_f1:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(weights_dir, f"{model_name}-{model_timestamp}-f1_{best_val_f1:.4f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with F1-score: {best_val_f1:.4f}")

    print("\nTraining complete!")

    save_and_plot_history(
        history=history,
        model_name=model_name,
        run_timestamp=run_timestamp,
        weights_dir=weights_dir
    )


SOURCE_DIRS = ['mac-merged', 'laptops-merged']
PROCESSED_DIR = 'data/processed'
WEIGHTS_DIR = 'models/weights'
MODEL_NAME = 'convnext'

NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_EPOCHS = 50
TARGET_AUG_COUNT = 1024
L1_LAMBDA = 1e-6
L2_LAMBDA = 1e-5
LR = 1e-4

if __name__ == '__main__':
    setup_dataset(
        source_base_dir='data',
        processed_base_dir=PROCESSED_DIR,
        source_class_names=SOURCE_DIRS,
        target_aug_count=TARGET_AUG_COUNT
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE
    )

    device = get_device()
    print(f"Selected device: {device}")

    model = create_dino_swin_classifier(num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=L2_LAMBDA)

    train_model(
        model=model,
        model_name=MODEL_NAME,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        weights_dir=WEIGHTS_DIR
    )
