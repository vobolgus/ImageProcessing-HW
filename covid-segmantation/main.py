import albumentations
import cv2

from data import visualize, prepare_data
from model import create_model
import scipy
import pandas as pd
from dataset import Dataset
from train import *
from plot import *

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# typed global used in prepare_data
batch_size: int = 0

SOURCE_SIZE: int = 512
TARGET_SIZE: int = 256
max_lr: float = 1e-3
epoch: int = 20
weight_decay: float = 1e-4

if __name__ == '__main__':
    train_images, train_masks, val_images, val_masks, test_images_medseg = prepare_data()

    train_augs = albumentations.Compose([
        albumentations.Rotate(limit=360, p=0.9, border_mode=cv2.BORDER_REPLICATE),
        albumentations.RandomSizedCrop((int(SOURCE_SIZE * 0.75), SOURCE_SIZE),
                                       (TARGET_SIZE, TARGET_SIZE),
                                       interpolation=cv2.INTER_NEAREST),
        albumentations.HorizontalFlip(p=0.5),

    ])

    val_augs = albumentations.Compose([
        albumentations.Resize(TARGET_SIZE, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    ])

    train_dataset = Dataset(train_images, train_masks, train_augs)
    val_dataset = Dataset(val_images, val_masks, val_augs)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    i, train_data = next(enumerate(train_dataloader))

    palette = [[0], [1], [2], [3]]
    mask_hot_encoded = mask_to_onehot(torch.unsqueeze(train_data[1], -1).numpy(), palette)
    # visualize(torch.unsqueeze(torch.squeeze(train_data[0],1),-1),mask_hot_encoded)
    visualize(train_data[0].permute(0, 2, 3, 1), mask_hot_encoded)

    model = create_model(device, num_classes=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_dataloader))

    history = fit(epoch, model, train_dataloader, val_dataloader, criterion, optimizer, sched)

    torch.save(model, 'Unet-efficientnet.pt')

    plot_loss(history)
    plot_score(history)
    plot_acc(history)

    image, mask = next(iter(val_dataloader))
    pred_mask, score, output = predict_image_mask_miou(model, image, mask)
    semantic_map = mask_to_onehot(torch.unsqueeze(mask, -1).numpy(), palette)

    # yellow is TP, red is FP, green is FN
    visualize(image, semantic_map, pred_batch=output.cpu())

    mob_miou = miou_score(model, val_dataloader)
    print("mob_miou:", mob_miou)

    del train_images
    del train_masks

    # test predictions

    image_batch = np.stack([val_augs(image=img)['image'] for img in test_images_medseg], axis=0)
    print(torch.from_numpy(image_batch).shape)
    print(image_batch[i].shape)
    # output = test_predict(model, torch.from_numpy(image_batch).permute(0, 3, 1,2))
    output = np.zeros((10, 256, 256, 4))
    for i in range(10):
        output[i] = test_predict(model, image_batch[i])
    print(output.shape)
    test_masks_prediction = output > 0.5
    visualize(image_batch, test_masks_prediction, num_samples=len(test_images_medseg))

    test_masks_prediction_original_size = scipy.ndimage.zoom(test_masks_prediction[..., :-2], (1, 2, 2, 1), order=0)
    print(test_masks_prediction_original_size.shape)

    frame = pd.DataFrame(
        data=np.stack(
            (np.arange(len(test_masks_prediction_original_size.ravel())),
             test_masks_prediction_original_size.ravel().astype(int)),
            axis=-1
        ),
        columns=['Id', 'Predicted']
    ) .set_index('Id')
    frame.to_csv('sub.csv')

