"""
label.py  —  Interactive manual labelling tool
-----------------------------------------------
Helps you create the initial ground-truth Excel file required for training.

The tool steps through a video, detects each unique LCD reading (via frame-diff),
shows you the cropped LCD in a window, and asks you to type the values.

Usage
-----
  python label.py --video my_video.mp4 --output my_labels.xlsx

Controls
--------
  Type  voltage,current  (e.g. "806.4,224.70")  then press OK to record.
  Leave blank and press OK to skip this frame.
  Press Cancel to save and quit.

Tips
----
  • You only need ~50-200 labelled frames for good accuracy.
  • Try to cover the full range of values seen in the video.
  • Re-running after quitting will append to the existing file.
"""

import argparse
import os
import tkinter as tk
from tkinter import simpledialog

import cv2
import numpy as np
import pandas as pd

from lcd_detector import LCDDetector


def label(args):
    detector = LCDDetector(
        hsv_lower=tuple(args.hsv_lower),
        hsv_upper=tuple(args.hsv_upper),
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if os.path.exists(args.output):
        df = pd.read_excel(args.output)
        start_frame = int(df["frame"].max()) + 1
        print(f"Resuming from frame {start_frame} (loaded {len(df)} existing labels)")
    else:
        df = pd.DataFrame(columns=["frame", "voltage", "current", "time"])
        start_frame = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    root = tk.Tk()
    root.withdraw()

    prev_gray: np.ndarray | None = None
    count = start_frame

    print("Labelling session started. Type values in the popup dialog. Cancel to quit.")

    while count < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        changed = False
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            changed = np.count_nonzero(thresh) > args.change_threshold

        prev_gray = gray

        if changed:
            # Let the display settle
            settle = min(count + args.settle_frames, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, settle)
            ret2, settled = cap.read()
            if not ret2:
                break

            patch = detector.detect(settled)
            display = settled if patch is None else cv2.cvtColor(
                cv2.resize(patch, (patch.shape[1] * 3, patch.shape[0] * 3),
                           interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_RGB2BGR,
            )

            win = "LCD (close dialog to continue)"
            cv2.namedWindow(win)
            cv2.moveWindow(win, 50, 200)
            cv2.imshow(win, display)
            cv2.waitKey(150)

            answer = simpledialog.askstring(
                "Label",
                f"Frame {settle}: voltage,current\n"
                "(e.g. 806.4,224.70 — leave blank to skip)",
                parent=root,
            )
            cv2.destroyAllWindows()

            if answer is None:   # Cancel → quit
                break

            answer = answer.strip()
            if answer:
                try:
                    v_str, c_str = answer.split(",")
                    v, c = float(v_str.strip()), float(c_str.strip())
                    new_row = pd.DataFrame([{
                        "frame":   settle,
                        "voltage": v,
                        "current": c,
                        "time":    settle / fps,
                    }])
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_excel(args.output, index=False)
                    print(f"  Saved frame {settle}: {v} V, {c} A")
                except Exception as e:
                    print(f"  Bad input ({e}), skipping.")

            count = settle + args.settle_frames
        else:
            count += 1

    print(f"\nDone. {len(df)} labels saved to {args.output}")
    cap.release()
    root.destroy()


def parse_args():
    p = argparse.ArgumentParser(description="Interactively label LCD readings in a video.")
    p.add_argument("--video",  required=True, help="Path to video file")
    p.add_argument("--output", required=True, help="Path to output Excel label file (.xlsx)")
    p.add_argument("--settle_frames", type=int, default=10,
                   help="Frames to skip after a change is detected (let LCD settle)")
    p.add_argument("--change_threshold", type=int, default=1000,
                   help="Pixel count threshold to detect a display change")
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
    label(parse_args())
