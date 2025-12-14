from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.model.freeze_utils import FreezeStrategy


DEFAULT_CLIP_MODEL = 'openai/clip-vit-base-patch32'


class CLIPClassifier(nn.Module):
    """CLIP Vision Encoder with a classification head.

    This uses only the image encoder from CLIP with a linear classification
    layer on top, following the same pattern as other classifiers in this codebase.
    """

    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.clip_model = CLIPModel.from_pretrained(model_name)

        # Get the vision encoder hidden size
        in_features = self.clip_model.config.vision_config.hidden_size

        # Create classification head on top of image embeddings
        self.head = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass accepting standard image tensor (B, C, H, W)."""
        # Get image features from CLIP vision encoder
        vision_outputs = self.clip_model.vision_model(pixel_values=x)
        # Use pooler output (CLS token representation)
        features = vision_outputs.pooler_output
        logits = self.head(features)
        return logits


def load_clip_from_weights(weights_path: str, num_classes: int) -> nn.Module:
    """Load a CLIP classifier from saved weights."""
    model = create_clip_classifier(num_classes=num_classes, freeze=None)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    return model


def create_clip_classifier(
        num_classes: int = 1000,
        pretrained: bool = True,
        freeze: Optional[FreezeStrategy] = None,
        model_name: str = DEFAULT_CLIP_MODEL,
) -> nn.Module:
    """
    Create a CLIP classifier matching the API of other model factories.

    Args:
        num_classes: Number of output classes
        pretrained: Ignored (CLIP always uses pretrained weights)
        freeze: Freezing strategy for the backbone
        model_name: HuggingFace model name

    Returns:
        CLIPClassifier model
    """
    model = CLIPClassifier(model_name, num_classes)

    # Apply freeze strategy
    if freeze is not None:
        _apply_clip_freeze(model, freeze)

    return model


def _apply_clip_freeze(model: CLIPClassifier, strategy: FreezeStrategy) -> None:
    """Apply freeze strategy to CLIP model."""
    # Always keep head trainable
    for param in model.head.parameters():
        param.requires_grad = True

    if strategy == FreezeStrategy.NO:
        return

    if strategy == FreezeStrategy.LAST:
        # Freeze entire vision backbone
        for param in model.clip_model.vision_model.parameters():
            param.requires_grad = False
        return

    if strategy == FreezeStrategy.PCT70:
        # Freeze first 70% of backbone parameters
        backbone_params = list(model.clip_model.vision_model.parameters())
        cutoff = int(len(backbone_params) * 0.7)
        for idx, param in enumerate(backbone_params):
            param.requires_grad = idx >= cutoff
        return

    if strategy == FreezeStrategy.LORA:
        # LoRA is applied separately via apply_lora function in main.py
        # Here we freeze the backbone; LoRA adapters will be added later
        for param in model.clip_model.vision_model.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    from torchvision import transforms

    image_path = "../../data/LaptopsVsMac/laptops/0.png"
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        exit(1)

    num_classes = 2

    model = create_clip_classifier(
        num_classes=num_classes,
        freeze=FreezeStrategy.PCT70,
    )
    model.eval()

    # Use standard ImageNet preprocessing (CLIP compatible)
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

    print(f"--- CLIP Model Info ---")
    print(f"Model: {DEFAULT_CLIP_MODEL}")
    print(f"Vision hidden size: {model.clip_model.config.vision_config.hidden_size}")
    print(f"Classifier head: {model.head}")
    print(f"\n--- Inference Results ---")
    print(f"Logits: {logits.tolist()}")
    print(f"Probabilities: {probabilities.tolist()}")
    print(f"Predicted class: {predicted_class_idx}")