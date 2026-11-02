"""
extract.py
----------
Run the trained model on a video and save all readings to an Excel file.

Usage
-----
  python extract.py --video my_video.mp4 --output readings.xlsx [options]

If --output already exists, the script resumes from the last processed frame,
so you can safely re-run it after interruption.

Validity bounds
---------------
Pass --vol_min, --vol_max, --cur_min, --cur_max to filter out obviously wrong
predictions (OCR noise, partially obscured display, etc.). Readings outside the
bounds are replaced by the most recent valid reading (forward-fill).

Output Excel format
--------------------
  frame | voltage | current | time
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from lcd_detector import LCDDetector
from model import MultiDigitNet, decode_prediction


# ─── Inference helpers ────────────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(model_path: str, device: torch.device) -> MultiDigitNet:
    model = MultiDigitNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(model, patch: np.ndarray, device: torch.device) -> tuple[float, float]:
    """Run one LCD patch through the model and return (voltage, current)."""
    img = TRANSFORM(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        preds = [torch.argmax(o, dim=1).item() for o in outputs]
    return decode_prediction(preds)


# ─── Main extraction loop ─────────────────────────────────────────────────────

def extract(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    model   = load_model(args.model, device)
    detector = LCDDetector(
        hsv_lower=tuple(args.hsv_lower),
        hsv_upper=tuple(args.hsv_upper),
    )

    # ── Resume support ──
    if os.path.exists(args.output):
        df = pd.read_excel(args.output)
        start_frame = int(df["frame"].max()) + 1
        last_vol = float(df["voltage"].iloc[-1])
        last_cur = float(df["current"].iloc[-1])
        print(f"Resuming from frame {start_frame} (existing output: {args.output})")
    else:
        df = pd.DataFrame(columns=["frame", "voltage", "current", "time"])
        start_frame = 0
        last_vol = float("nan")
        last_cur = float("nan")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps

    if start_frame >= total_frames:
        print("All frames already processed.")
        return

    print(
        f"Video: {args.video}  |  {total_frames} frames  |  "
        f"{fps:.2f} fps  |  {duration_s:.1f} s"
    )
    print(
        f"Validity bounds  — voltage: [{args.vol_min}, {args.vol_max}]  "
        f"current: [{args.cur_min}, {args.cur_max}]"
    )
    print(f"Processing every {args.frame_step} frame(s) …\n")

    new_rows: list[dict] = []

    for count in range(start_frame, total_frames, args.frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        ret, frame = cap.read()
        if not ret:
            break

        patch = detector.detect(frame)
        vol, cur = last_vol, last_cur

        if patch is not None:
            v, c = predict(model, patch, device)
            if not np.isnan(v) and args.vol_min <= v <= args.vol_max:
                vol = v
            if not np.isnan(c) and args.cur_min <= c <= args.cur_max:
                cur = c

        last_vol, last_cur = vol, cur
        new_rows.append({
            "frame":   count,
            "voltage": vol,
            "current": cur,
            "time":    count / fps,
        })

        if len(new_rows) % 200 == 0:
            pct = 100.0 * (count - start_frame) / max(1, total_frames - start_frame)
            print(f"  [{pct:5.1f}%] frame {count:6d}  V={vol:.1f}  A={cur:.2f}")

    cap.release()

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df["voltage"] = df["voltage"].ffill().bfill()
        df["current"] = df["current"].ffill().bfill()
        df.to_excel(args.output, index=False)
        print(f"\nSaved {len(new_rows)} new rows → {args.output}")
    else:
        print("No new data extracted.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract LCD readings from video using a trained model."
    )
    p.add_argument("--video",   required=True,  help="Path to input video file")
    p.add_argument("--output",  required=True,  help="Path to output Excel file (.xlsx)")
    p.add_argument("--model",   default="lcd_ocr_model.pth", help="Path to trained model weights")
    p.add_argument("--frame_step", type=int, default=5,
                   help="Process every Nth frame (default 5 — fast enough, captures all changes)")
    # Validity bounds
    p.add_argument("--vol_min", type=float, default=0.0,   help="Minimum plausible voltage")
    p.add_argument("--vol_max", type=float, default=9999.0,help="Maximum plausible voltage")
    p.add_argument("--cur_min", type=float, default=0.0,   help="Minimum plausible current")
    p.add_argument("--cur_max", type=float, default=9999.0,help="Maximum plausible current")
    # LCD detection
    p.add_argument(
        "--hsv_lower", type=int, nargs=3, default=[40, 50, 50],
        metavar=("H", "S", "V"),
        help="Lower HSV bound for LCD border colour"
    )
    p.add_argument(
        "--hsv_upper", type=int, nargs=3, default=[90, 255, 255],
        metavar=("H", "S", "V"),
        help="Upper HSV bound for LCD border colour"
    )
    return p.parse_args()


if __name__ == "__main__":
    extract(parse_args())
