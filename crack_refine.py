"""
Integration-layer crack refinement (Member 4).

Member 2's detector (crack_detector.py, kept unmodified) is deliberately sensitive. On real
underwater scenes it also fires on gravel, marine growth, ROV/diver equipment and sensor
noise. A crack is a *long, thin, relatively isolated* dark line, whereas texture is *dense and
short*. This module post-filters M2's mask using three classical, explainable rules:

    1. length   : skeleton length  >= min_len_frac * image diagonal
    2. thinness : mean width <= max_width_frac * image diagonal (reject blobs)
    3. mesh test: skeleton junctions per 100 px of length <= max_junctions. Texture, marine growth
                  and rusty surfaces form branching *meshes* (7-28 junctions/100 px measured on the
                  sample set); isolated cracks are mostly 0-5.
    4. isolation: local density of detected pixels *belonging to OTHER components* around
                  this component must be low (reject dense clusters = texture). The
                  component's own pixels are excluded, otherwise any isolated thick crack
                  would be penalised by its own width.
"""
from __future__ import annotations
import cv2
import numpy as np
from skimage.morphology import skeletonize

DEFAULT_REFINE = {"min_len_frac": 0.05, "max_width_frac": 0.015, "max_density": 0.10, "max_junctions": 6.0, "enabled": True}


def refine_crack_mask(mask, params=None):
    p = dict(DEFAULT_REFINE)
    p.update(params or {})
    if not p["enabled"] or not np.any(mask):
        return mask.copy(), {"removed": 0, "kept": int(bool(np.any(mask)))}
    h, w = mask.shape
    diag = float(np.hypot(h, w))
    win = max(int(diag * 0.08) | 1, 15)
    binary = (mask > 0).astype(np.float32)
    density = cv2.blur(binary, (win, win))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    out = np.zeros_like(mask)
    kept = removed = 0
    pad = win
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        comp = labels[y:y + bh, x:x + bw] == i
        skel = skeletonize(comp)
        length = float(skel.sum())
        width = float(2 * dist[y:y + bh, x:x + bw][skel].mean()) if length else 0.0
        # density of OTHER components around this one (own pixels removed)
        y0, y1, x0, x1 = max(y - pad, 0), min(y + bh + pad, h), max(x - pad, 0), min(x + bw + pad, w)
        own = np.zeros((y1 - y0, x1 - x0), np.float32)
        own[y - y0:y - y0 + bh, x - x0:x - x0 + bw][comp] = 1.0
        others = density[y0:y1, x0:x1] - cv2.blur(own, (win, win))
        dens = float(np.clip(others[y - y0:y - y0 + bh, x - x0:x - x0 + bw][comp], 0, None).mean())
        nb = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - skel
        junc_rate = 100.0 * float(((nb >= 3) & skel).sum()) / max(length, 1.0)
        ok = (length >= p["min_len_frac"] * diag and width <= p["max_width_frac"] * diag
              and dens <= p["max_density"] and junc_rate <= p["max_junctions"])
        if ok:
            out[y:y + bh, x:x + bw][comp] = 255
            kept += 1
        else:
            removed += 1
    return out, {"removed": removed, "kept": kept}
