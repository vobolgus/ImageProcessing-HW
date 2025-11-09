import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from lightning_module import CovidSegmenter
from lightning_datamodule import CovidDataModule
from freeze_utils import FreezeStrategy

batch_size: int = 32

SOURCE_SIZE: int = 512
TARGET_SIZE: int = 256
MAX_LR: float = 1e-3
EPOCHS: int = 20
WEIGHT_DECAY: float = 1e-4
TRAIN_BATCH_SIZE: int = 24

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    datamodule = CovidDataModule(
        train_batch_size=TRAIN_BATCH_SIZE,
        source_size=SOURCE_SIZE,
        target_size=TARGET_SIZE
    )

    model = CovidSegmenter(
        num_classes=4,
        max_lr=MAX_LR,
        weight_decay=WEIGHT_DECAY,
        freeze_strategy=FreezeStrategy.PCT70
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_miou',
        dirpath='checkpoints',
        filename='best_model-{epoch:02d}-{val_miou:.3f}',
        save_top_k=1,
        mode='max',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=7,
        mode='min'
    )

    csv_logger = CSVLogger(save_dir="logs/")

    trainer = pl.Trainer(
        accelerator='mps',
        devices=1,
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=csv_logger
    )

    print("Starting PyTorch Lightning training...")
    trainer.fit(model, datamodule=datamodule)

    print("Training complete.")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")