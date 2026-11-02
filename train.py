"""
train.py
--------
Train the MultiDigitNet on frames paired with manually transcribed readings.

Usage
-----
  python train.py --video my_video.mp4 --excel my_data.xlsx [options]

The script reads (frame_number, voltage, current) rows from the Excel sheet,
seeks to each frame in the video, extracts the LCD patch, and trains the model.

The trained weights are saved to  lcd_ocr_model.pth  when training finishes.

Customising for your LCD
-------------------------
  1. Adjust --hsv_lower / --hsv_upper to match your LCD border colour.
  2. Adjust --vol_decimals and --cur_decimals if your display uses a different
     number of decimal places.
  3. Add more labelled rows to your Excel sheet to improve accuracy.

Excel format expected
----------------------
  Columns: frame | voltage | current
  Any extra columns are silently ignored.
"""

import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import MultiDigitNet, encode_label
from lcd_detector import LCDDetector


# ─── Dataset ─────────────────────────────────────────────────────────────────

class LCDDataset(Dataset):
    def __init__(
        self,
        excel_path: str,
        video_path: str,
        detector: LCDDetector,
        augment: bool = True,
    ):
        df = pd.read_excel(excel_path).dropna(subset=["frame", "voltage", "current"])
        cap = cv2.VideoCapture(video_path)

        self.images: list[np.ndarray] = []
        self.labels: list[list[int]] = []

        aug_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.93, 1.07)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        plain_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform = aug_tf if augment else plain_tf

        print(f"Loading {len(df)} labelled frames …")
        for _, row in df.iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
            ret, frame = cap.read()
            if not ret:
                continue
            patch = detector.detect(frame)
            if patch is None:
                continue
            self.images.append(patch)
            self.labels.append(encode_label(float(row["voltage"]), float(row["current"])))

        cap.release()
        print(f"Dataset ready – {len(self.images)} valid samples.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, lbl


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    detector = LCDDetector(
        hsv_lower=tuple(args.hsv_lower),
        hsv_upper=tuple(args.hsv_upper),
    )
    full_ds = LCDDataset(args.excel, args.video, detector, augment=True)
    if len(full_ds) == 0:
        print("ERROR: No samples could be extracted. Check your HSV bounds and Excel file.")
        return

    n_train = int(0.85 * len(full_ds))
    n_val = len(full_ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MultiDigitNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 3, gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = sum(criterion(outputs[i], lbls[:, i]) for i in range(10))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # ── validate ──
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                ok = torch.ones(imgs.size(0), dtype=torch.bool, device=device)
                for i in range(10):
                    ok &= torch.argmax(outputs[i], dim=1) == lbls[:, i]
                correct += ok.sum().item()
                total += imgs.size(0)

        acc = 100.0 * correct / total if total else 0.0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={total_loss/len(train_loader):.4f}  "
            f"exact_match={acc:.1f}%"
        )
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.model_out)

    print(f"\nBest exact-match accuracy: {best_acc:.1f}%")
    print(f"Model saved to: {args.model_out}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train the LCD OCR model on your labelled frames."
    )
    p.add_argument("--video",     required=True,  help="Path to the training video file")
    p.add_argument("--excel",     required=True,  help="Path to the labelled Excel file (frame | voltage | current)")
    p.add_argument("--model_out", default="lcd_ocr_model.pth", help="Output path for the trained model weights")
    p.add_argument("--epochs",    type=int,   default=80,   help="Number of training epochs")
    p.add_argument("--batch_size",type=int,   default=32,   help="Mini-batch size")
    p.add_argument("--lr",        type=float, default=5e-4, help="Initial learning rate")
    p.add_argument(
        "--hsv_lower", type=int, nargs=3, default=[40, 50, 50],
        metavar=("H", "S", "V"),
        help="Lower HSV threshold for LCD border detection (default: 40 50 50 = green)"
    )
    p.add_argument(
        "--hsv_upper", type=int, nargs=3, default=[90, 255, 255],
        metavar=("H", "S", "V"),
        help="Upper HSV threshold for LCD border detection (default: 90 255 255)"
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
