"""CLIP Contrastive classifier using image-text embedding similarity.

This implementation uses the original CLIP approach:
- Text prompts "a photo of {class}" are encoded
- Images are encoded
- Similarity between image and text embeddings determines the class

No classification head is used - pure contrastive learning.
"""
from enum import Enum
from typing import Optional, List

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizerFast
from peft import LoraConfig, get_peft_model


DEFAULT_CLIP_MODEL = 'openai/clip-vit-base-patch32'


class CLIPFreezeStrategy(Enum):
    """Freeze strategies specific to CLIP contrastive model."""
    ZERO_SHOT = "zero_shot"           # No training, frozen model
    PCT30 = "pct30"                   # Train 30% of last layers (both encoders)
    FULL = "full"                     # Train all weights (ImageEncoder + TextEncoder)
    VISION_ONLY = "vision_only"       # Train only ImageEncoder
    TEXT_ONLY = "text_only"           # Train only TextEncoder
    LORA = "lora"                     # Use LoRA adapters


class CLIPContrastiveClassifier(nn.Module):
    """CLIP classifier using image-text embedding similarity (no classification head).

    Works by:
    1. Pre-computing text embeddings for all class labels ("a photo of {class}")
    2. At inference, computing image embeddings
    3. Finding most similar text embedding via dot product
    """

    def __init__(
        self,
        clip_model: CLIPModel,
        tokenizer: CLIPTokenizerFast,
        labels: List[str],
    ):
        super().__init__()
        self.model = clip_model
        self.tokenizer = tokenizer
        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        # Generate and store label embeddings
        self._update_label_embeddings()

    def _get_device(self) -> torch.device:
        """Get current device of the model."""
        return next(self.model.parameters()).device

    @torch.no_grad()
    def _update_label_embeddings(self) -> None:
        """Generate text embeddings for all class labels."""
        device = self._get_device()

        # Create prompts for each label
        prompts = [f"a photo of {label}" for label in self.labels]

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # Get text features
        text_features = self.model.get_text_features(**inputs)

        # Normalize embeddings
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # Register as buffer (not a parameter, but moves with model)
        self.register_buffer('labels_embeddings', text_features)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores between images and label embeddings.

        Args:
            pixel_values: Image tensor of shape (B, C, H, W)

        Returns:
            Similarity logits of shape (B, num_classes)
        """
        # Get image features
        image_features = self.model.get_image_features(pixel_values)

        # Normalize
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Compute similarity (using CLIP's learned temperature)
        logit_scale = self.model.logit_scale.exp()
        logits = torch.matmul(image_features, self.labels_embeddings.T) * logit_scale

        return logits

    def update_labels(self, labels: List[str]) -> None:
        """Update class labels and regenerate embeddings."""
        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self._update_label_embeddings()


def _freeze_params(module: nn.Module, freeze_top_percent: float = 1.0) -> None:
    """Freeze a percentage of parameters starting from the top.

    Args:
        module: PyTorch module to freeze
        freeze_top_percent: Fraction of parameters to freeze (from the beginning)
    """
    all_params = list(module.parameters())
    cutoff = int(len(all_params) * freeze_top_percent)
    for idx, param in enumerate(all_params):
        if idx < cutoff:
            param.requires_grad = False


def _apply_clip_contrastive_freeze(
    model: CLIPContrastiveClassifier,
    strategy: CLIPFreezeStrategy
) -> None:
    """Apply freeze strategy to CLIP contrastive model."""

    if strategy == CLIPFreezeStrategy.ZERO_SHOT:
        # Freeze everything
        for param in model.model.parameters():
            param.requires_grad = False
        return

    if strategy == CLIPFreezeStrategy.FULL:
        # Train everything
        for param in model.model.parameters():
            param.requires_grad = True
        return

    if strategy == CLIPFreezeStrategy.VISION_ONLY:
        # Freeze text encoder, train vision encoder
        for param in model.model.text_model.parameters():
            param.requires_grad = False
        for param in model.model.vision_model.parameters():
            param.requires_grad = True
        # Also train the visual projection
        if hasattr(model.model, 'visual_projection'):
            for param in model.model.visual_projection.parameters():
                param.requires_grad = True
        return

    if strategy == CLIPFreezeStrategy.TEXT_ONLY:
        # Freeze vision encoder, train text encoder
        for param in model.model.vision_model.parameters():
            param.requires_grad = False
        for param in model.model.text_model.parameters():
            param.requires_grad = True
        # Also train the text projection
        if hasattr(model.model, 'text_projection'):
            for param in model.model.text_projection.parameters():
                param.requires_grad = True
        return

    if strategy == CLIPFreezeStrategy.PCT30:
        # Freeze first 70% of both encoders (train last 30%)
        _freeze_params(model.model.text_model, freeze_top_percent=0.7)
        _freeze_params(model.model.vision_model, freeze_top_percent=0.7)
        # Train projections
        if hasattr(model.model, 'visual_projection'):
            for param in model.model.visual_projection.parameters():
                param.requires_grad = True
        if hasattr(model.model, 'text_projection'):
            for param in model.model.text_projection.parameters():
                param.requires_grad = True
        return

    if strategy == CLIPFreezeStrategy.LORA:
        # LoRA is applied separately - freeze all here
        for param in model.model.parameters():
            param.requires_grad = False
        return


def create_clip_contrastive_classifier(
    labels: List[str],
    freeze: Optional[CLIPFreezeStrategy] = None,
    model_name: str = DEFAULT_CLIP_MODEL,
) -> CLIPContrastiveClassifier:
    """Create a CLIP contrastive classifier.

    Args:
        labels: List of class label strings
        freeze: Freeze strategy to apply
        model_name: HuggingFace model name

    Returns:
        CLIPContrastiveClassifier model
    """
    # Load CLIP model and tokenizer
    clip_model = CLIPModel.from_pretrained(model_name)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

    # Create classifier wrapper
    model = CLIPContrastiveClassifier(clip_model, tokenizer, labels)

    # Apply freeze strategy
    if freeze is not None:
        _apply_clip_contrastive_freeze(model, freeze)

    return model


def apply_lora_to_clip_contrastive(model: CLIPContrastiveClassifier) -> CLIPContrastiveClassifier:
    """Apply LoRA adapters to CLIP contrastive model.

    Uses LoraConfig(r=64, lora_alpha=64, target_modules='all').

    Args:
        model: CLIPContrastiveClassifier to apply LoRA to

    Returns:
        Model with LoRA adapters applied to the internal CLIP model
    """
    config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules='all-linear',  # Apply to all linear layers
        lora_dropout=0.1,
    )

    # Apply LoRA to the internal CLIP model
    model.model = get_peft_model(model.model, config)

    return model


def load_clip_contrastive_from_weights(
    weights_path: str,
    labels: List[str],
    model_name: str = DEFAULT_CLIP_MODEL,
) -> CLIPContrastiveClassifier:
    """Load a CLIP contrastive classifier from saved weights."""
    model = create_clip_contrastive_classifier(
        labels=labels,
        freeze=None,
        model_name=model_name,
    )
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    return model


def print_trainable_parameters(model: nn.Module) -> None:
    """Print statistics about trainable parameters."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {(trainable_params / 10**6):.4f}M || "
        f"All params: {(all_params / 10**6):.4f}M || "
        f"Trainable%: {100 * trainable_params / all_params:.2f}%"
    )


if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image

    # Example usage
    labels = ["laptop", "mac"]

    print("Creating CLIP Contrastive Classifier...")
    model = create_clip_contrastive_classifier(
        labels=labels,
        freeze=CLIPFreezeStrategy.ZERO_SHOT,
    )
    model.eval()

    print("\n--- Zero-shot (frozen) ---")
    print_trainable_parameters(model)

    # Test with different strategies
    for strategy in CLIPFreezeStrategy:
        print(f"\n--- {strategy.value} ---")
        test_model = create_clip_contrastive_classifier(
            labels=labels,
            freeze=strategy,
        )
        if strategy == CLIPFreezeStrategy.LORA:
            test_model = apply_lora_to_clip_contrastive(test_model)
        print_trainable_parameters(test_model)

    # Test inference
    print("\n--- Testing Inference ---")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dummy image
    dummy_image = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        logits = model(dummy_image)

    probs = torch.softmax(logits, dim=-1)
    print(f"Labels: {labels}")
    print(f"Logits: {logits.tolist()}")
    print(f"Probabilities: {probs.tolist()}")
    print(f"Predicted: {labels[logits.argmax(-1).item()]}")