# ======================================================================
#  Classical CV pipeline for peri-implant bone-level extraction
#  Author: A. Huerta Moncho, 2025
# ======================================================================

# --- 1. Standard imports ------------------------------------------------
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from typing import Dict, Any

# --- 2. Helper functions -------------------------------------------------
def adaptive_threshold_mean(image: np.ndarray,
                            block_size: int = 15,
                            C: int = 10) -> np.ndarray:
    """Mean adaptive threshold using integral images."""

    image = image.astype(np.float32)
    rows, cols = image.shape
    integral   = cv2.integral(image)
    half       = block_size // 2
    output     = np.zeros_like(image, dtype=np.uint8)
    for y in range(rows):
        y0, y1 = max(0, y-half), min(rows, y+half+1)
        for x in range(cols):
            x0, x1 = max(0, x-half), min(cols, x+half+1)
            area = (y1-y0)*(x1-x0)
            summ = (integral[y1, x1] - integral[y0, x1]
                  - integral[y1, x0] + integral[y0, x0])
            output[y, x] = 255 if image[y, x] > (summ/area - C) else 0
    return output

def log_transform(gray: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Contrast-enhancing logarithmic transform."""
    g = gray.astype(np.float32) / 255.0
    log_im = gain * np.log1p(g)
    log_im = cv2.normalize(log_im, None, 0, 255, cv2.NORM_MINMAX)
    return log_im.astype(np.uint8)

def preprocess_gray(gray: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    return (log_transform(gray, params["log_gain"])
            if params["prep_mode"] == "log" else gray)

# --- 3. Pipeline core ----------------------------------------------------
def run_pipeline(img_bgr: np.ndarray, p: Dict[str, Any]) -> Dict[str, Any]:
    gray = preprocess_gray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), p)

    smooth = cv2.medianBlur(gray, p["median_ksize"])

    adapt_bin = adaptive_threshold_mean(gray, p["block_size"], p["C"])

    thresholds = threshold_multiotsu(smooth, classes=p["n_classes"])
    labeled = np.digitize(smooth, bins=thresholds)

    # --- small-object removal ---
    cleaned = labeled.copy()
    for cls in range(p["n_classes"]):
        mask = (labeled == cls).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:  # only background
            continue
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        for i in range(1, n):
            if i != largest and stats[i, cv2.CC_STAT_AREA] < p["min_area"]:
                cleaned[lab == i] = -1

    # --- hole filling ---
    invalid = (cleaned == -1)
    while invalid.any():
        dilated = cv2.dilate((cleaned != -1).astype(np.uint8),
                             np.ones((3, 3), np.uint8))
        border = dilated & invalid
        for y, x in zip(*np.where(border)):
            neigh = cleaned[max(y-1, 0):y+2, max(x-1, 0):x+2]
            vals = neigh[neigh != -1]
            if vals.size:
                cleaned[y, x] = np.bincount(vals).argmax()
        invalid = (cleaned == -1)

    # --- masks -----------------------------------------------------------
    mask_impl = (cleaned == p["implante_clase"])
    mask_edge = (~mask_impl) & (adapt_bin == 255)

    # --- conditioned gradient -------------------------------------------
    edges = np.zeros_like(cleaned, np.uint8)
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        shifted_lab   = np.roll(cleaned, (dy, dx),    (0,1))
        shifted_valid = np.roll(mask_edge, (dy, dx),  (0,1))
        cond = mask_edge & shifted_valid & (cleaned != shifted_lab)
        edges |= cond.astype(np.uint8)
    edges *= 255

    return dict(gray=gray, adaptive=adapt_bin, labeled=labeled,
                cleaned=cleaned, edges=edges, thresholds=thresholds)

