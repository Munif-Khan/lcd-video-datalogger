"""
model.py
--------
Shared model definition and helper utilities for the LCD Video DataLogger.

The MultiDigitNet is a ResNet-18 backbone with 10 independent classification
heads — one per character position on the LCD screen.

Label encoding
--------------
  Voltage  : up to 4 significant digits + 1 decimal → stored as 5 digit-slots
              e.g.  806.4  →  "8064"  →  [10, 8, 0, 6, 4]   (10 = "blank")
  Current  : up to 3 significant digits + 2 decimals → 5 digit-slots
              e.g.  224.70 →  "22470" →  [10, 2, 2, 4, 7]
  Combined : [v0, v1, v2, v3, v4, c0, c1, c2, c3, c4]  (length 10)

Each head outputs logits for classes 0-10 (digits 0-9, plus 10 = blank/pad).
"""

import torch
import torch.nn as nn
from torchvision import models
import numpy as np


# ─── Model ───────────────────────────────────────────────────────────────────

class MultiDigitNet(nn.Module):
    """Multi-head ResNet-18 for simultaneous 10-digit LCD reading."""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18()
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        # 10 heads × 11 classes (digits 0-9 + blank class 10)
        self.heads = nn.ModuleList([nn.Linear(num_ftrs, 11) for _ in range(10)])

    def forward(self, x):
        features = self.backbone(x)
        return [head(features) for head in self.heads]


# ─── Encoding / Decoding ──────────────────────────────────────────────────────

def encode_label(voltage: float, current: float) -> list[int]:
    """
    Convert a (voltage, current) reading into a 10-element integer label.

    Parameters
    ----------
    voltage : float  e.g. 806.4
    current : float  e.g. 224.70

    Returns
    -------
    list[int] of length 10, values in 0-10  (10 = blank padding)
    """
    vol_str = f"{voltage:.1f}".replace(".", "")   # e.g. "8064"
    cur_str = f"{current:.2f}".replace(".", "")   # e.g. "22470"
    vol_digits = [10] * (5 - len(vol_str)) + [int(c) for c in vol_str]
    cur_digits = [10] * (5 - len(cur_str)) + [int(c) for c in cur_str]
    return vol_digits + cur_digits


def decode_prediction(
    preds: list[int],
    vol_scale: float = 10.0,
    cur_scale: float = 100.0,
) -> tuple[float, float]:
    """
    Convert a 10-element predicted label back to (voltage, current).

    Parameters
    ----------
    preds     : list of 10 ints (output from model argmax)
    vol_scale : divisor to recover voltage float  (default 10  → 1 decimal place)
    cur_scale : divisor to recover current float  (default 100 → 2 decimal places)

    Returns
    -------
    (voltage, current) as floats, or (nan, nan) if decoding fails.
    """
    vol_str = "".join(str(d) for d in preds[:5] if d != 10)
    cur_str = "".join(str(d) for d in preds[5:] if d != 10)
    try:
        vol = float(vol_str) / vol_scale if vol_str else float("nan")
        cur = float(cur_str) / cur_scale if cur_str else float("nan")
    except ValueError:
        vol, cur = float("nan"), float("nan")
    return vol, cur
