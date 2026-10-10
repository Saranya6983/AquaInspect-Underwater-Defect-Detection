"""
Combined severity scoring and colour-coded damage map (Member 3's formula, made scale-safe).

Member 3's original formula:  score = 0.6 * corrosion_percent + 0.4 * crack_length
used a *raw pixel* crack length (hard-coded to 10 in the batch script), so the crack term
was either meaningless or exploded with image resolution. Here crack length is normalised by
image size so the score is resolution independent and both terms live on a 0-100 scale:

    crack_density = crack_length_px / sqrt(W*H)         (1.0 = crack as long as the image side)
    crack_index   = min(100, 50 * crack_density)         (2 image-sides of crack => 100)
    score         = 0.6 * corrosion_percent_index + 0.4 * crack_index
    corrosion_idx = min(100, 100 * corrosion_percent / 40)   (>=40% of the surface rusted => 100)

Levels (Member 3's cut-offs retained): Low < 20 <= Medium < 50 <= High.
"""
from __future__ import annotations
import cv2
import numpy as np

W_CORR, W_CRACK = 0.6, 0.4
LOW_MAX, MED_MAX = 20.0, 50.0
CORR_SATURATION_PCT = 40.0
CRACK_SATURATION_DENSITY = 2.0
LEVEL_COLORS_BGR = {"Low": (80, 200, 80), "Medium": (0, 190, 255), "High": (50, 50, 230)}


def classify(score, low=LOW_MAX, med=MED_MAX):
    return "Low" if score < low else ("Medium" if score < med else "High")


def severity(corrosion_percent, crack_length_px, shape):
    h, w = shape[:2]
    density = crack_length_px / max(np.sqrt(h * w), 1.0)
    crack_idx = min(100.0, 100.0 * density / CRACK_SATURATION_DENSITY)
    corr_idx = min(100.0, 100.0 * corrosion_percent / CORR_SATURATION_PCT)
    score = W_CORR * corr_idx + W_CRACK * crack_idx
    return {"corrosion_percent": round(float(corrosion_percent), 2),
            "crack_density": round(float(density), 3),
            "corrosion_index": round(float(corr_idx), 2),
            "crack_index": round(float(crack_idx), 2),
            "severity_score": round(float(score), 2),
            "severity_level": classify(score)}


def damage_map(image, crack_mask, corrosion_mask, cells=24, alpha=0.45):
    """Grid-based damage map. Each cell gets Low/Medium/High from local crack+corrosion density.
    Returns (annotated image, per-cell level grid, % of area per level)."""
    h, w = image.shape[:2]
    cs = max(int(round(max(h, w) / cells)), 8)
    gy, gx = int(np.ceil(h / cs)), int(np.ceil(w / cs))
    cm = (corrosion_mask > 0).astype(np.float32)
    kk = (crack_mask > 0).astype(np.float32)
    kk = cv2.dilate(kk, np.ones((3, 3), np.uint8))
    grid = np.zeros((gy, gx), dtype=np.int8)   # 0 none, 1 low, 2 med, 3 high
    overlay = image.copy()
    for i in range(gy):
        for j in range(gx):
            ys, xs = slice(i * cs, min((i + 1) * cs, h)), slice(j * cs, min((j + 1) * cs, w))
            corr_f, crack_f = cm[ys, xs].mean(), kk[ys, xs].mean()
            s = 100 * min(1.0, W_CORR * min(1.0, corr_f / 0.6) + W_CRACK * min(1.0, crack_f / 0.10))
            if s < 5:
                continue
            lvl = 1 if s < 20 else (2 if s < 50 else 3)
            grid[i, j] = lvl
            col = LEVEL_COLORS_BGR[("Low", "Medium", "High")[lvl - 1]]
            cv2.rectangle(overlay, (xs.start, ys.start), (xs.stop - 1, ys.stop - 1), col, -1)
    out = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    # thin grid outlines on damaged cells
    for i, j in zip(*np.nonzero(grid)):
        col = LEVEL_COLORS_BGR[("Low", "Medium", "High")[grid[i, j] - 1]]
        cv2.rectangle(out, (j * cs, i * cs), (min((j + 1) * cs, w) - 1, min((i + 1) * cs, h) - 1), col, 1)
    total = grid.size
    pct = {n: round(100.0 * float((grid == k).sum()) / total, 2) for k, n in ((1, "Low"), (2, "Medium"), (3, "High"))}
    return out, grid, pct
