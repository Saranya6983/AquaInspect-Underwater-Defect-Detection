"""
Member 4 - integrated backend pipeline.

    input -> validate -> enhance (M1) -> crack detect+measure (M2) -> corrosion (M3)
          -> severity (M3, normalised) -> damage map -> result dict

All inter-module data exchange happens through the dict returned by `run_inspection`.
"""
from __future__ import annotations
import time
import cv2
import numpy as np

from . import enhancement as enh
from .crack_detector import CrackDetector
from .crack_refine import refine_crack_mask
from . import corrosion as corr
from .severity import severity, damage_map

MIN_SIDE = 48


class InvalidImageError(ValueError):
    """Raised for unreadable / unusable input images."""


def load_image(source):
    """Accept a path, raw bytes (e.g. Streamlit upload) or an ndarray; return a 3-channel BGR uint8."""
    try:
        if isinstance(source, (bytes, bytearray)):
            img = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_UNCHANGED)
        elif isinstance(source, str):
            img = cv2.imread(source, cv2.IMREAD_UNCHANGED)
        elif isinstance(source, np.ndarray):
            img = source.copy()
        else:
            raise InvalidImageError(f"Unsupported input type: {type(source).__name__}")
    except cv2.error as e:
        raise InvalidImageError(f"Could not decode image: {e}")
    if img is None or img.size == 0:
        raise InvalidImageError("File is not a readable image.")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if min(img.shape[:2]) < MIN_SIDE:
        raise InvalidImageError(f"Image too small ({img.shape[1]}x{img.shape[0]}); need at least {MIN_SIDE}px per side.")
    return img


def run_inspection(source, enhance_config=None, enhance=True, refine=True, refine_params=None,
                   crack_params=None, max_side=1280, pixel_to_mm=None, with_map=True):
    """Run the whole AquaInspect pipeline on one image. Returns a dict (images are BGR/uint8)."""
    t = {}
    t0 = time.perf_counter()
    original = load_image(source)
    original = enh.resize_max(original, max_side)
    t["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    enhanced = enh.enhance(original, enhance_config) if enhance else original.copy()
    t["enhance"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    det = CrackDetector(pixel_to_mm_ratio=pixel_to_mm, **(crack_params or {}))
    cres = det.process(enhanced)
    raw_mask = cres["mask"]
    if refine:
        mask, rinfo = refine_crack_mask(raw_mask, {**(refine_params or {}), "enabled": True})
    else:
        mask, rinfo = raw_mask, {"removed": 0, "kept": cres["metrics"]["num_crack_segments"]}
    skeleton = det.compute_skeleton(mask)
    cmetrics = det.measure_cracks(mask, skeleton)
    t["crack"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    hsv = corr.convert_color_spaces(enhanced)
    cmask = corr.remove_noise(hsv, corr.clean_mask(corr.detect_corrosion(hsv)))
    c_px, c_pct = corr.calculate_area(cmask)
    t["corrosion"] = time.perf_counter() - t0

    sev = severity(c_pct, cmetrics["crack_length_px"], enhanced.shape)
    out = {"original": original, "enhanced": enhanced, "crack_mask": mask, "crack_mask_raw": raw_mask,
           "crack_skeleton": skeleton, "corrosion_mask": cmask,
           "crack_overlay": det.create_visual_overlay(enhanced, mask, skeleton),
           "crack_metrics": cmetrics, "refine_info": rinfo,
           "corrosion_metrics": {"corrosion_pixels": int(c_px), "corrosion_percent": round(float(c_pct), 2)},
           "severity": sev, "shape": enhanced.shape[:2]}
    if with_map:
        out["damage_map"], out["damage_grid"], out["damage_pct"] = damage_map(enhanced, mask, cmask)
    out["corrosion_overlay"] = corrosion_overlay(enhanced, cmask)
    out["timings"] = {k: round(v, 3) for k, v in t.items()}
    return out


def corrosion_overlay(image, cmask, color=(0, 110, 255), alpha=0.5):
    ov = image.copy()
    ov[cmask > 0] = color
    return cv2.addWeighted(ov, alpha, image, 1 - alpha, 0)


def summary_row(name, r):
    """Flatten one result into a CSV-friendly dict."""
    cm, s = r["crack_metrics"], r["severity"]
    return {"Image": name, "Width": r["shape"][1], "Height": r["shape"][0],
            "Crack_Detected": "YES" if cm["crack_detected"] else "NO",
            "Crack_Segments": cm["num_crack_segments"], "Crack_Length_px": cm["crack_length_px"],
            "Mean_Width_px": cm["mean_width_px"], "Max_Width_px": cm["max_width_px"],
            "Crack_Area_px": cm["crack_area_px"], "Corrosion_%": s["corrosion_percent"],
            "Crack_Index": s["crack_index"], "Severity_Score": s["severity_score"],
            "Severity_Level": s["severity_level"], "Time_s": round(sum(r["timings"].values()), 2)}
