"""Batch-process a folder of images with the integrated pipeline (no UI).

Usage: python scripts/run_batch.py [--input data/sample_images] [--output results/batch]
Writes per-image enhanced / mask / overlay / damage-map images and results/batch/batch_results.csv
"""
import argparse, glob, json, os, sys, warnings
import cv2, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aquainspect.pipeline import run_inspection, summary_row, InvalidImageError
from aquainspect import enhancement as E

ap = argparse.ArgumentParser()
ap.add_argument("--input", default=os.path.join(ROOT, "data", "sample_images"))
ap.add_argument("--output", default=os.path.join(ROOT, "results", "batch"))
a = ap.parse_args()
cfgp = os.path.join(ROOT, "results", "selected_config.json")
cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else E.DEFAULT_CONFIG
for sub in ("enhanced", "crack_masks", "corrosion_masks", "damage_maps"):
    os.makedirs(os.path.join(a.output, sub), exist_ok=True)
rows = []
for f in sorted(glob.glob(os.path.join(a.input, "*"))):
    if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue
    stem = os.path.splitext(os.path.basename(f))[0]
    try:
        r = run_inspection(f, enhance_config=cfg)
    except InvalidImageError as e:
        print("SKIP", f, e); continue
    cv2.imwrite(os.path.join(a.output, "enhanced", stem + ".jpg"), r["enhanced"])
    cv2.imwrite(os.path.join(a.output, "crack_masks", stem + ".png"), r["crack_mask"])
    cv2.imwrite(os.path.join(a.output, "corrosion_masks", stem + ".png"), r["corrosion_mask"])
    cv2.imwrite(os.path.join(a.output, "damage_maps", stem + ".jpg"), r["damage_map"])
    rows.append(summary_row(os.path.basename(f), r)); print(rows[-1]["Image"], rows[-1]["Severity_Level"], rows[-1]["Severity_Score"], flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(a.output, "batch_results.csv"), index=False)
print(df.Severity_Level.value_counts().to_string())
