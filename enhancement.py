"""
Member 1 - Underwater image enhancement (importable module).

Member 1's original script (member_work/member1_enhancement/enhancement.py) used one
fixed chain: LAB-CLAHE -> Non-Local-Means denoising. This module keeps that chain
(`member1_original`) and adds the alternative methods needed for the stage-by-stage
comparison described in the project abstract.

Stages and the methods available at each stage:
    color     : none | gray_world | white_patch | red_comp
    dehaze    : none | dcp | udcp
    denoise   : none | gaussian | median | bilateral | nlm
    contrast  : none | he | clahe
All functions take and return uint8 BGR images.
"""
from __future__ import annotations
import cv2
import numpy as np

COLOR_METHODS = ["none", "gray_world", "white_patch", "red_comp"]
DEHAZE_METHODS = ["none", "dcp", "udcp"]
DENOISE_METHODS = ["none", "gaussian", "median", "bilateral", "nlm"]
CONTRAST_METHODS = ["none", "he", "clahe"]

# Chain chosen by downstream detection score on the synthetic benchmark (scripts/run_benchmark.py; see docs/DOCUMENTATION.md)
DEFAULT_CONFIG = {"color": "red_comp", "dehaze": "none", "denoise": "gaussian", "contrast": "clahe"}
# Member 1's original fixed chain
MEMBER1_CONFIG = {"color": "none", "dehaze": "none", "denoise": "nlm", "contrast": "clahe", "order": ("contrast", "denoise")}
STAGE_ORDER = ("color", "dehaze", "denoise", "contrast")


# ---------------------------------------------------------------- colour correction
def gray_world(img):
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0) + 1e-6
    f *= means.mean() / means
    return np.clip(f, 0, 255).astype(np.uint8)


def white_patch(img, pct=99):
    f = img.astype(np.float32)
    hi = np.percentile(f.reshape(-1, 3), pct, axis=0) + 1e-6
    f *= 255.0 / hi
    return np.clip(f, 0, 255).astype(np.uint8)


def red_comp(img):
    """Red-channel compensation (water absorbs red first) followed by gray-world."""
    f = img.astype(np.float32) / 255.0
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    r = r + (g.mean() - r.mean()) * (1.0 - r) * g
    out = np.dstack([b, g, np.clip(r, 0, 1)])
    return gray_world((out * 255).astype(np.uint8))


# ---------------------------------------------------------------- dehazing
def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))


def _guided(guide, src, r=30, eps=1e-3):
    mg, ms = _box(guide, r), _box(src, r)
    cov = _box(guide * src, r) - mg * ms
    var = _box(guide * guide, r) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return _box(a, r) * guide + _box(b, r)


def _dehaze(img, channels, omega=0.85, t0=0.25, patch=15):
    f = img.astype(np.float32) / 255.0
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(f[..., channels].min(axis=2), k)
    n = max(int(dark.size * 0.001), 1)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    A = f.reshape(-1, 3)[idx].mean(0).clip(0.4, 1.0)
    norm = f / A
    t = 1.0 - omega * cv2.erode(norm[..., channels].min(axis=2), k)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = np.clip(_guided(gray, t.astype(np.float32)), t0, 1.0)
    J = (f - A) / t[..., None] + A
    return np.clip(J * 255, 0, 255).astype(np.uint8)


def dcp(img):
    """Dark Channel Prior (He et al.) using all three channels."""
    return _dehaze(img, [0, 1, 2])


def udcp(img):
    """Underwater DCP - dark channel from blue/green only (red is unreliable underwater)."""
    return _dehaze(img, [0, 1])


# ---------------------------------------------------------------- denoising
def d_gaussian(img): return cv2.GaussianBlur(img, (5, 5), 0)
def d_median(img): return cv2.medianBlur(img, 5)
def d_bilateral(img): return cv2.bilateralFilter(img, 9, 50, 50)
def d_nlm(img): return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


# ---------------------------------------------------------------- contrast
def he(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def clahe(img, clip=2.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


_FUNCS = {
    "color": {"gray_world": gray_world, "white_patch": white_patch, "red_comp": red_comp},
    "dehaze": {"dcp": dcp, "udcp": udcp},
    "denoise": {"gaussian": d_gaussian, "median": d_median, "bilateral": d_bilateral, "nlm": d_nlm},
    "contrast": {"he": he, "clahe": clahe},
}


def apply_stage(img, stage, method):
    """Apply one method of one stage. method == 'none' returns the image unchanged."""
    if method in (None, "none"):
        return img.copy()
    if method not in _FUNCS[stage]:
        raise ValueError(f"Unknown {stage} method '{method}'")
    return _FUNCS[stage][method](img)


def enhance(img, config=None):
    """Run the enhancement chain. `config` maps stage -> method (see DEFAULT_CONFIG)."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config or {})
    order = cfg.get("order", STAGE_ORDER)
    out = img.copy()
    for stage in order:
        out = apply_stage(out, stage, cfg.get(stage, "none"))
    return out


def resize_max(img, max_side=1280):
    """Down-scale very large photos (keeps NLM/skeletonization fast). Never up-scales."""
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
