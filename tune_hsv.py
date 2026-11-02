"""
tune_hsv.py
-----------
Interactive HSV tuner — helps you find the right colour bounds for your LCD
border so that lcd_detector.py can reliably locate the screen.

Usage
-----
  python tune_hsv.py --video my_video.mp4
  python tune_hsv.py --image frame.jpg        # or from a still image

Controls
--------
  Six trackbars (H_low, S_low, V_low, H_high, S_high, V_high) let you
  adjust the threshold live. The right panel shows the binary mask; the left
  panel shows the detected bounding box drawn on the original frame.

  Press S to save the current values to hsv_config.txt
  Press Q or Esc to quit.
"""

import argparse
import sys
import cv2
import numpy as np


def nothing(_):
    pass


def main(args):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Cannot read image: {args.image}")
        frames = [frame]
        is_video = False
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"Cannot open video: {args.video}")
        is_video = True

    WIN = "HSV Tuner  [S=save  Q=quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 500)

    # Trackbars
    for name, init, maxv in [
        ("H_low", 40, 179), ("S_low", 50, 255), ("V_low", 50, 255),
        ("H_high", 90, 179), ("S_high", 255, 255), ("V_high", 255, 255),
    ]:
        cv2.createTrackbar(name, WIN, init, maxv, nothing)

    print("Adjust the trackbars until the LCD border is highlighted white in the mask.")
    print("Press S to save values, Q/Esc to quit.")

    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        else:
            frame = frames[0].copy()

        hl = cv2.getTrackbarPos("H_low",  WIN)
        sl = cv2.getTrackbarPos("S_low",  WIN)
        vl = cv2.getTrackbarPos("V_low",  WIN)
        hh = cv2.getTrackbarPos("H_high", WIN)
        sh = cv2.getTrackbarPos("S_high", WIN)
        vh = cv2.getTrackbarPos("V_high", WIN)

        lower = np.array([hl, sl, vl], dtype=np.uint8)
        upper = np.array([hh, sh, vh], dtype=np.uint8)

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # Draw bounding box
        vis = frame.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([
            cv2.resize(vis, (640, 480)),
            cv2.resize(mask_bgr, (640, 480)),
        ])
        cv2.putText(combined,
                    f"lower=({hl},{sl},{vl})  upper=({hh},{sh},{vh})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WIN, combined)

        key = cv2.waitKey(30 if is_video else 50) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            cfg = f"--hsv_lower {hl} {sl} {vl} --hsv_upper {hh} {sh} {vh}"
            with open("hsv_config.txt", "w") as f:
                f.write(cfg + "\n")
            print(f"Saved → hsv_config.txt\n  {cfg}")

    cv2.destroyAllWindows()
    if is_video:
        cap.release()


def parse_args():
    p = argparse.ArgumentParser(description="Interactive HSV threshold tuner for LCD detection.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="Path to a video file")
    g.add_argument("--image", help="Path to a still image (JPG/PNG)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
