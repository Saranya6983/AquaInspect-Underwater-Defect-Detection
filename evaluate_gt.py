"""Evaluate detection against MANUALLY ANNOTATED ground truth.

Folder convention (data/ground_truth/):
    <image_stem>_crack.png      white (>127) = crack pixels
    <image_stem>_corrosion.png  white (>127) = corrosion pixels
Either file may be missing (that defect is then skipped for that image).
Masks are resized to the analysed image size (nearest neighbour).
"""
from __future__ import annotations
import glob, os
import cv2
import pandas as pd
from .pipeline import run_inspection
from .metrics import seg_metrics

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _gt(path, shape):
    if not os.path.exists(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return ((m > 127) * 255).astype("uint8")


def annotated_images(images_dir, gt_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(images_dir, "*"))):
        if f.lower().endswith(IMG_EXT):
            stem = os.path.splitext(os.path.basename(f))[0]
            if any(os.path.exists(os.path.join(gt_dir, f"{stem}_{k}.png")) for k in ("crack", "corrosion")):
                out.append(f)
    return out


def evaluate_folder(images_dir, gt_dir, enhance_config=None, crack_tol=3, progress=None, **kw):
    files = annotated_images(images_dir, gt_dir)
    rows = []
    for i, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        r = run_inspection(f, enhance_config=enhance_config, with_map=False, **kw)
        row = {"Image": os.path.basename(f)}
        for kind, mask, tol in (("crack", r["crack_mask"], crack_tol), ("corrosion", r["corrosion_mask"], 0)):
            g = _gt(os.path.join(gt_dir, f"{stem}_{kind}.png"), r["shape"])
            if g is not None:
                for k, v in seg_metrics(mask, g, tol=tol).items():
                    row[f"{kind}_{k}"] = round(v, 4)
        rows.append(row)
        if progress:
            progress((i + 1) / len(files))
    return pd.DataFrame(rows)
