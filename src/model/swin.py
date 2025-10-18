from typing import Optional, Tuple
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights  # type: ignore


def _get_default_weights(pretrained: bool):
    if pretrained:
        return Swin_B_Weights.DEFAULT
    return None


def create_swin_classifier(
        num_classes: int = 1000,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        weights: Optional[object] = None,
) -> Tuple[nn.Module, Optional[object]]:
    used_weights = weights if weights is not None else _get_default_weights(pretrained)

    model = models.swin_b(weights=used_weights)

    in_features = model.head.in_features
    if num_classes != model.head.out_features:
        model.head = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(model.head.weight, std=0.02)
        if model.head.bias is not None:
            nn.init.zeros_(model.head.bias)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if name.startswith("head"):
                param.requires_grad = True
            else:
                param.requires_grad = False
    return model, used_weights


if __name__ == '__main__':

    image_path = "../../data/mac-merged/0.png"
    image = Image.open(image_path).convert('RGB')

    model, weights = create_swin_classifier(num_classes=2, pretrained=True, freeze_backbone=True)
    model.eval()

    preprocess = weights.transforms()

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()

    print(f"Image: {image_path}")
    print(f"Image size: {image.size}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Class probabilities: {probabilities.tolist()}")
