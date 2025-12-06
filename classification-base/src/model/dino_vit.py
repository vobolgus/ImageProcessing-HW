from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel

from src.model.freeze_utils import FreezeStrategy


# DINOv3 model from HuggingFace (Meta's self-supervised vision model)
DEFAULT_DINOV3_MODEL = 'facebook/dinov3-vits16-pretrain-lvd1689m'


class DINOv3Classifier(nn.Module):
    """DINOv3 backbone with a classification head."""

    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.base_model = AutoModel.from_pretrained(model_name)

        in_features = self.base_model.config.hidden_size

        self.head = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass accepting standard image tensor (B, C, H, W)."""
        outputs = self.base_model(pixel_values=x)
        features = outputs.pooler_output
        logits = self.head(features)
        return logits


def load_dinov3_from_weights(weights_path: str, num_classes: int) -> nn.Module:
    """Load a DINOv3 classifier from saved weights."""
    model = create_dinov3_classifier(num_classes=num_classes, freeze=None)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    return model


def create_dinov3_classifier(
        num_classes: int = 1000,
        pretrained: bool = True,
        freeze: Optional[FreezeStrategy] = None,
        model_name: str = DEFAULT_DINOV3_MODEL,
) -> nn.Module:
    """
    Create a DINOv3 classifier matching the API of other model factories.

    Args:
        num_classes: Number of output classes
        pretrained: Ignored (DINOv3 always uses pretrained weights)
        freeze: Freezing strategy for the backbone
        model_name: HuggingFace model name

    Returns:
        DINOv3Classifier model
    """
    model = DINOv3Classifier(model_name, num_classes)

    # Apply freeze strategy
    if freeze is not None:
        _apply_dinov3_freeze(model, freeze)

    return model


def _apply_dinov3_freeze(model: DINOv3Classifier, strategy: FreezeStrategy) -> None:
    """Apply freeze strategy to DINOv3 model."""
    # Always keep head trainable
    for param in model.head.parameters():
        param.requires_grad = True

    if strategy == FreezeStrategy.NO:
        return

    if strategy == FreezeStrategy.LAST:
        # Freeze entire backbone
        for param in model.base_model.parameters():
            param.requires_grad = False
        return

    if strategy == FreezeStrategy.PCT70:
        # Freeze first 70% of backbone parameters
        backbone_params = list(model.base_model.parameters())
        cutoff = int(len(backbone_params) * 0.7)
        for idx, param in enumerate(backbone_params):
            param.requires_grad = idx >= cutoff


if __name__ == "__main__":
    from torchvision import transforms

    image_path = "../../data/LaptopsVsMac/laptops/0.png"
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        exit(1)

    num_classes = 2

    model = create_dinov3_classifier(
        num_classes=num_classes,
        freeze=FreezeStrategy.PCT70,
    )
    model.eval()

    # Use standard ImageNet preprocessing (DINOv3 compatible)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)

    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class_idx = logits.argmax(-1).item()

    print(f"--- DINOv3 Model Info ---")
    print(f"Model: {DEFAULT_DINOV3_MODEL}")
    print(f"Hidden size: {model.base_model.config.hidden_size}")
    print(f"Classifier head: {model.head}")
    print(f"\n--- Inference Results ---")
    print(f"Logits: {logits.tolist()}")
    print(f"Probabilities: {probabilities.tolist()}")
    print(f"Predicted class: {predicted_class_idx}")