from enum import Enum
from typing import Optional, Tuple

import torch.nn as nn


class FreezeStrategy(Enum):
    """Enum describing backbone freezing strategies."""
    NO = "no"              # train all weights
    LAST = "last"          # freeze entire backbone, train classifier head only
    PCT70 = "freeze_70"    # freeze first 70% of backbone params (train 30%)
    LORA = "lora"          # use LoRA adapters instead of full fine-tuning


def _parse_freeze_strategy(freeze: Optional[FreezeStrategy]) -> FreezeStrategy:
    return freeze if isinstance(freeze, FreezeStrategy) else FreezeStrategy.NO


def apply_freeze(model: nn.Module, classifier_prefixes: Tuple[str, ...], strategy: Optional[FreezeStrategy]) -> None:
    """Freeze backbone parameters according to the provided strategy.

    Classifier parameters (whose names start with any of classifier_prefixes)
    are always kept trainable.
    """

    strategy: FreezeStrategy = _parse_freeze_strategy(freeze=strategy)

    # Ensure classifier is trainable
    for name, p in model.named_parameters():
        if name.startswith(classifier_prefixes):
            p.requires_grad = True

    if strategy == FreezeStrategy.NO:
        return

    backbone_params = [(n, p) for n, p in model.named_parameters() if not n.startswith(classifier_prefixes)]

    if strategy == FreezeStrategy.LAST:
        for _, p in backbone_params:
            p.requires_grad = False
        return

    if strategy == FreezeStrategy.PCT70:
        total = len(backbone_params)
        cutoff = int(total * 0.7)
        for idx, (_, p) in enumerate(backbone_params):
            p.requires_grad = False if idx < cutoff else True
        return

    if strategy == FreezeStrategy.LORA:
        # LoRA is applied separately via apply_lora function
        # Here we just freeze all backbone params; LoRA adapters will be added later
        for _, p in backbone_params:
            p.requires_grad = False
        return


def apply_lora(model: nn.Module) -> nn.Module:
    """Apply LoRA adapters to the model.

    Uses peft library with LoraConfig(r=64, lora_alpha=64, target_modules='all').

    Args:
        model: The model to apply LoRA to

    Returns:
        Model wrapped with LoRA adapters
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules='all-linear',  # Apply to all linear layers
        lora_dropout=0.1,
    )

    lora_model = get_peft_model(model, config)
    return lora_model


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
