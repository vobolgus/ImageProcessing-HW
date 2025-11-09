import torch
import os
from monai.networks.nets import SwinUNETR
from torchinfo import summary
import segmentation_models_pytorch as smp

def create_model(device, num_classes=4, pretrained_weights_path="models/model_swinvit.pt"):
    """
    Создает SOTA 3D-модель (SwinUNETR) для сегментации и загружает
    предобученные веса для fine-tuning.

    Args:
        device (torch.device): Устройство (GPU/CPU), на котором будет модель.
        num_classes (int): Количество выходных классов (в вашем случае 4).
        pretrained_weights_path (str): Путь к файлу весов 'model_swinvit.pt'.

    Returns:
        torch.nn.Module: Готовая к обучению SOTA-модель.
    """

    model = smp.Unet(
        # 'tu-' означает, что мы берем модель из timm
        encoder_name="tu-swin_tiny_patch4_window7_224",
        encoder_weights="imagenet",
        in_channels=1,
        classes=num_classes,
    )

    model.to(device)

    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device, num_classes=4)
    print("\nModel created successfully:")
    print(f"Model is on device: {next(model.parameters()).device}")

    # --- 2. Вызываем summary ---
    # Указываем (batch_size, channels, H, W)
    # (H и W взяты из вашего TARGET_SIZE = 256 в main.py)
    batch_size = 16
    input_channels = 1
    image_size = 256

    print("\n--- Model Summary ---")
    summary(model, input_size=(batch_size, input_channels, image_size, image_size))