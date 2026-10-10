"""
Evaluation metrics.

Enhancement : UIQM, UCIQE (no-reference), PSNR / SSIM (full-reference), noise sigma estimate.
Detection   : IoU, precision, recall, F1 (against a ground-truth mask).

NOTE: UIQM here follows the standard definition (Panetta et al., 2016) but UIConM uses a
simplified log-AME block contrast rather than full PLIP arithmetic, so absolute values are
not comparable with numbers published elsewhere - use it for *relative* comparison.
"""
from __future__ import annotations
import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ------------------------------------------------------------------ UIQM
def _trim_mean_var(x, alpha=0.1):
    x = np.sort(x.ravel())
    n = len(x)
    lo, hi = int(alpha * n), int((1 - alpha) * n)
    t = x[lo:hi] if hi > lo else x
    return t.mean(), t.var()


def _eme(x, block=8):
    h, w = x.shape
    bh, bw = max(h // block, 1), max(w // block, 1)
    x = x[: bh * block, : bw * block].astype(np.float64)
    blocks = x.reshape(bh, block, bw, block).swapaxes(1, 2).reshape(-1, block * block)
    mx, mn = blocks.max(1), blocks.min(1)
    ok = (mx > 0) & (mn > 0)
    if not ok.any():
        return 0.0
    return float(2.0 / (bh * bw) * np.sum(np.log(mx[ok] / mn[ok])))


def uiqm(bgr):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    # UICM
    mu_rg, v_rg = _trim_mean_var(r - g)
    mu_yb, v_yb = _trim_mean_var((r + g) / 2 - b)
    uicm = -0.0268 * np.hypot(mu_rg, mu_yb) + 0.1586 * np.sqrt(v_rg + v_yb)
    # UISM
    uism = 0.0
    for ch, wgt in ((r, 0.299), (g, 0.587), (b, 0.114)):
        sx = cv2.Sobel(ch, cv2.CV_64F, 1, 0)
        sy = cv2.Sobel(ch, cv2.CV_64F, 0, 1)
        edge = np.hypot(sx, sy)
        uism += wgt * _eme(edge * ch)
    # UIConM (simplified log-AME)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    k = 8
    bh, bw = max(h // k, 1), max(w // k, 1)
    blk = gray[: bh * k, : bw * k].reshape(bh, k, bw, k).swapaxes(1, 2).reshape(-1, k * k)
    mx, mn = blk.max(1), blk.min(1)
    v = (mx - mn) / (mx + mn + 1e-6)
    uiconm = float(-np.mean(v * np.log(v + 1e-6)))
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


# ------------------------------------------------------------------ UCIQE
def uciqe(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab = rgb2lab(rgb)
    L, a, b = lab[..., 0] / 100.0, lab[..., 1] / 100.0, lab[..., 2] / 100.0
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = chroma.std()
    lv = np.sort(L.ravel())
    n = max(int(0.01 * len(lv)), 1)
    con_l = lv[-n:].mean() - lv[:n].mean()
    sat = chroma / (np.sqrt(chroma ** 2 + L ** 2) + 1e-6)
    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * sat.mean())


# ------------------------------------------------------------------ full-reference
def _match(a, b):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return a, b


def psnr(ref, img):
    ref, img = _match(ref, img)
    return float(peak_signal_noise_ratio(ref, img, data_range=255))


def ssim(ref, img):
    ref, img = _match(ref, img)
    return float(structural_similarity(ref, img, channel_axis=2, data_range=255))


def noise_sigma(bgr):
    """Immerkaer fast noise-level estimate (lower = cleaner)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = g.shape
    s = np.abs(cv2.filter2D(g, -1, k)[1:-1, 1:-1]).sum()
    return float(s * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))


def enhancement_scores(original, enhanced):
    return {"UIQM": uiqm(enhanced), "UCIQE": uciqe(enhanced), "NoiseSigma": noise_sigma(enhanced),
            "PSNR_vs_input": psnr(original, enhanced), "SSIM_vs_input": ssim(original, enhanced)}


# ------------------------------------------------------------------ detection
def seg_metrics(pred, gt, tol=0):
    """IoU (exact) and precision/recall/F1 (optionally with `tol` px boundary tolerance,
    which is the fair choice for 1-5 px wide cracks)."""
    p, g = pred > 0, gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = inter / union if union else 1.0
    if tol > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        g_d = cv2.dilate(g.astype(np.uint8), k) > 0
        p_d = cv2.dilate(p.astype(np.uint8), k) > 0
        tp_p, tp_r = np.logical_and(p, g_d).sum(), np.logical_and(g, p_d).sum()
    else:
        tp_p = tp_r = inter
    prec = tp_p / p.sum() if p.sum() else (1.0 if not g.sum() else 0.0)
    rec = tp_r / g.sum() if g.sum() else (1.0 if not p.sum() else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"IoU": float(iou), "Precision": float(prec), "Recall": float(rec), "F1": float(f1)}
