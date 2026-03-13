import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from torch.cuda.amp import autocast, GradScaler
from dataset import SegDataset

train_losses = []
train_ious = []
train_accs = []

def compute_iou(pred, mask, num_classes):
    ious = []

    pred = pred.view(-1)
    mask = mask.view(-1)

    for cls in range(num_classes):

        pred_inds = pred == cls
        target_inds = mask == cls

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    if len(ious) == 0:
        return 0

    return sum(ious) / len(ious)


def main():

    print("Starting training script...")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    NUM_CLASSES = 6
    BATCH_SIZE = 2
    EPOCHS = 15
    LR = 5e-5

    print("Loading datasets...")

    train_dataset = SegDataset(
        "dataset/Offroad_Segmentation_Training_Dataset/train/Color_Images",
        "dataset/Offroad_Segmentation_Training_Dataset/train/Segmentation"
    )

    val_dataset = SegDataset(
        "dataset/Offroad_Segmentation_Training_Dataset/val/Color_Images",
        "dataset/Offroad_Segmentation_Training_Dataset/val/Segmentation"
    )

    print(f"Train images: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print("Loading SegFormer model...")

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    scaler = GradScaler()

    print("Training started...")

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0
        train_iou = 0

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        for batch_idx, (images, masks) in enumerate(train_loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            with autocast():

                outputs = model(pixel_values=images)
                logits = outputs.logits

                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

                loss = criterion(logits, masks)

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            batch_iou = compute_iou(preds, masks, NUM_CLASSES)
            train_iou += batch_iou

            if batch_idx % 50 == 0:
                print(
                    f"Batch {batch_idx}/{len(train_loader)}  "
                    f"Loss: {loss.item():.4f}  IoU: {batch_iou:.4f}"
                )

        avg_loss = train_loss / len(train_loader)
        avg_iou = train_iou / len(train_loader)

        print(f"\nEpoch {epoch+1} Summary")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Mean IoU: {avg_iou:.4f}")

    torch.save(model.state_dict(), "segformer_offroad_model.pth")
    print("\nModel saved as segformer_offroad_model.pth")


if __name__ == "__main__":
    main()