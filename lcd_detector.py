"""
lcd_detector.py
---------------
Locates and crops the LCD panel from a single video frame.

The detector uses HSV colour thresholding to find the bright green/backlit
border of the LCD enclosure, then returns the cropped region resized to a
fixed (128 × 64) patch ready for the model.

Tunable Parameters (passed as constructor arguments)
----------------------------------------------------
hsv_lower   : lower HSV bound for the LCD border colour (default green)
hsv_upper   : upper HSV bound for the LCD border colour
min_area    : minimum contour area in pixels to be considered a valid LCD
output_size : (width, height) to resize the cropped LCD patch to
"""

import cv2
import numpy as np


class LCDDetector:
    def __init__(
        self,
        hsv_lower=(40, 50, 50),
        hsv_upper=(90, 255, 255),
        min_area: int = 1000,
        output_size: tuple[int, int] = (128, 64),
    ):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area
        self.output_size = output_size  # (width, height)

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Detect and crop the LCD panel from *frame* (BGR, uint8).

        Returns
        -------
        np.ndarray of shape (H, W, 3) in RGB, dtype uint8
            — resized to self.output_size — or None if not found.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        crop = frame[y : y + h, x : x + w]
        crop = cv2.resize(crop, self.output_size)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of *frame* with the detected LCD bounding box drawn."""
        vis = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= self.min_area:
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(
                    vis, "LCD", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                )
        return vis
