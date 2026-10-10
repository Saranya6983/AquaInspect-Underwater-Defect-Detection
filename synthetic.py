"""
Synthetic underwater inspection benchmark (controlled, with exact ground truth).

Real annotated underwater crack/corrosion data was not available, so this module generates:
  * a clean concrete-pier or steel-pipe surface (procedural texture)
  * cracks (jagged random walks, optional branches) and rust patches with EXACT masks
  * distractors with no ground truth - green marine growth and soft shadows (-> false positives)
  * underwater degradation: wavelength-dependent attenuation (red lost first), veiling-light haze,
    colour cast, blur and sensor noise

This allows true full-reference PSNR/SSIM (degraded->enhanced vs. clean) and IoU/precision/recall/F1.
It is a *proxy* for manual annotation - results must not be presented as real-world accuracy.
"""
from __future__ import annotations
import cv2
import numpy as np


def _smooth_noise(rng, h, w, sigma):
    n = rng.standard_normal((h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return (n - n.mean()) / (n.std() + 1e-6)


def make_surface(rng, h, w, kind="concrete"):
    tex = 0.5 * _smooth_noise(rng, h, w, 40) + 0.35 * _smooth_noise(rng, h, w, 8) + 0.25 * _smooth_noise(rng, h, w, 1.5)
    if kind == "concrete":
        base = np.array([150, 158, 162], np.float32)            # BGR light warm-grey
        img = base[None, None, :] + 14 * tex[..., None]
        for _ in range(rng.integers(30, 80)):                    # pits / air voids
            cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(1, 3)), (95, 100, 104), -1)
    else:                                                        # steel pipe with cylindrical shading
        y = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        shade = 0.55 + 0.45 * np.sqrt(np.clip(1 - y ** 2, 0, 1))
        base = np.array([110, 112, 116], np.float32)
        img = base[None, None, :] * shade[..., None] + 7 * tex[..., None]
    return np.clip(img, 0, 255).astype(np.float32)


def add_cracks(rng, img, n=(1, 3)):
    h, w = img.shape[:2]
    gt = np.zeros((h, w), np.uint8)

    def walk(x, y, ang, length, width):
        pts = [(x, y)]
        for _ in range(length // 6):
            ang += rng.normal(0, 0.35)
            x += 6 * np.cos(ang); y += 6 * np.sin(ang)
            pts.append((x, y))
        pts = np.array(pts, np.int32).reshape(-1, 1, 2)
        cv2.polylines(gt, [pts], False, 255, int(width))
        return pts
    for _ in range(int(rng.integers(n[0], n[1] + 1))):
        x, y = rng.uniform(0.1 * w, 0.9 * w), rng.uniform(0.1 * h, 0.9 * h)
        L = int(rng.integers(int(0.35 * w), int(0.8 * w)))
        wd = int(rng.integers(2, 5))
        pts = walk(x, y, rng.uniform(0, 2 * np.pi), L, wd)
        if rng.random() < 0.6:                                    # branch
            i = int(rng.integers(len(pts) // 3, 2 * len(pts) // 3 + 1)) if len(pts) > 3 else 0
            bx, by = pts[min(i, len(pts) - 1), 0]
            walk(float(bx), float(by), rng.uniform(0, 2 * np.pi), L // 3, max(wd - 1, 1))
    soft = cv2.GaussianBlur(gt, (0, 0), 0.8).astype(np.float32) / 255.0
    img *= (1 - 0.72 * soft[..., None])
    return img, gt


def add_rust(rng, img, max_patches=3):
    h, w = img.shape[:2]
    gt = np.zeros((h, w), np.uint8)
    rust = np.zeros_like(img)
    for _ in range(int(rng.integers(0, max_patches + 1))):
        field = _smooth_noise(rng, h, w, rng.uniform(12, 25))
        cx, cy = rng.uniform(0.2 * w, 0.8 * w), rng.uniform(0.2 * h, 0.8 * h)
        yy, xx = np.mgrid[0:h, 0:w]
        field = field - ((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (rng.uniform(0.12, 0.28) * w) ** 2)
        m = (field > 0.1).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        gt = cv2.bitwise_or(gt, m)
    if gt.any():
        grain = _smooth_noise(rng, h, w, 1.5)
        col = np.array([28, 78, 165], np.float32)                 # BGR rust orange-brown (H~12 in OpenCV HSV)
        rust = col[None, None, :] * (0.8 + 0.18 * grain[..., None]) 
        soft = cv2.GaussianBlur(gt, (0, 0), 1.2).astype(np.float32)[..., None] / 255.0
        img = img * (1 - soft) + rust * soft
    return img, gt


def add_distractors(rng, img):
    h, w = img.shape[:2]
    # green marine growth (no GT)
    for _ in range(int(rng.integers(0, 3))):
        f = _smooth_noise(rng, h, w, rng.uniform(8, 18)); m = (f > 1.0).astype(np.float32)
        m = cv2.GaussianBlur(m, (0, 0), 2)[..., None]
        img = img * (1 - 0.8 * m) + np.array([60, 130, 55], np.float32) * 0.8 * m
    # soft shadow (no GT)
    if rng.random() < 0.7:
        x0 = int(rng.integers(0, w)); sh = np.zeros((h, w), np.float32); cv2.circle(sh, (x0, int(rng.integers(0, h))), int(0.25 * w), 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), 30)[..., None]
        img = img * (1 - 0.6 * sh)
    return img


def underwater(rng, clean, strength=1.0):
    f = clean.astype(np.float32) / 255.0
    depth = rng.uniform(1.5, 3.5) * strength
    atten = np.array([0.10, 0.45, 0.80], np.float32)             # B,G,R attenuation coefficients
    yy = np.linspace(0, 1, f.shape[0], dtype=np.float32)[:, None, None]
    t = np.exp(-atten[None, None, :] * depth * (0.7 + 0.6 * yy))  # farther at top -> hazier
    veil = np.array([0.62, 0.55, 0.12], np.float32)               # blue-green veiling light (BGR)
    haze = np.exp(-0.35 * strength * (1.2 - yy))
    out = f * t * haze + veil * (1 - haze) * 0.55
    out = cv2.GaussianBlur(out, (0, 0), 0.9)
    out += rng.normal(0, 0.025, out.shape).astype(np.float32)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def make_sample(seed, h=384, w=512):
    rng = np.random.default_rng(seed)
    kind = "concrete" if seed % 2 == 0 else "steel"
    img = make_surface(rng, h, w, kind)
    img, rust_gt = add_rust(rng, img)
    img, crack_gt = add_cracks(rng, img)
    img = add_distractors(rng, img)
    clean = np.clip(img, 0, 255).astype(np.uint8)
    return {"clean": clean, "degraded": underwater(rng, clean), "crack_gt": crack_gt, "rust_gt": rust_gt, "kind": kind, "seed": seed}


def make_benchmark(n=24, seed0=100):
    return [make_sample(seed0 + i) for i in range(n)]
