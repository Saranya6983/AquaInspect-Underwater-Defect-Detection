"""Stage-by-stage enhancement comparison (UIQM / UCIQE / PSNR / SSIM / noise) + method selection.

Usage:  python scripts/run_enhancement_eval.py [--max-side 800]
Outputs (results/): enhancement_per_image.csv, enhancement_summary.csv, selected_enhancement.json,
                    enhancement_comparison.png, enhancement_samples.png
"""
import argparse, glob, json, os, sys, time, warnings
import cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aquainspect import enhancement as E, metrics as M

STAGES = {"color": E.COLOR_METHODS, "dehaze": E.DEHAZE_METHODS,
          "denoise": E.DENOISE_METHODS, "contrast": E.CONTRAST_METHODS}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-side", type=int, default=800)
    a = ap.parse_args()
    out = os.path.join(ROOT, "results"); os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "data", "sample_images", "*")))
    rows = []
    for f in files:
        img = E.resize_max(cv2.imread(f), a.max_side); name = os.path.basename(f)
        base = M.enhancement_scores(img, img); base.update(Stage="baseline", Method="original", Image=name, Time_s=0)
        rows.append(base)
        for st, methods in STAGES.items():
            for m in methods:
                if m == "none": continue
                t = time.perf_counter(); o = E.apply_stage(img, st, m); dt = time.perf_counter() - t
                r = M.enhancement_scores(img, o); r.update(Stage=st, Method=m, Image=name, Time_s=dt); rows.append(r)
        print("done", name, flush=True)
    df = pd.DataFrame(rows).replace([np.inf], np.nan)
    df.to_csv(os.path.join(out, "enhancement_per_image.csv"), index=False)
    summ = df.groupby(["Stage", "Method"], sort=False)[["UIQM", "UCIQE", "NoiseSigma", "PSNR_vs_input", "SSIM_vs_input", "Time_s"]].mean().round(4).reset_index()
    summ.to_csv(os.path.join(out, "enhancement_summary.csv"), index=False)

    # ---- selection: mean rank over stage-specific criteria
    sel, rank_tbl = {}, []
    base = summ[summ.Stage == "baseline"].iloc[0]
    for st in STAGES:
        s = summ[summ.Stage == st].copy()
        if st != "denoise":   # allow "skip this stage" as a candidate
            b = base.copy(); b["Method"] = "none"; b["Stage"] = st; s = pd.concat([s, b.to_frame().T], ignore_index=True)
            for c in ["UIQM", "UCIQE", "NoiseSigma", "PSNR_vs_input", "SSIM_vs_input", "Time_s"]: s[c] = s[c].astype(float)
            crit = [("UIQM", False), ("UCIQE", False)]
        else:
            crit = [("NoiseSigma", True), ("SSIM_vs_input", False)]
        s["rank"] = sum(s[c].rank(ascending=asc) for c, asc in crit) / len(crit)
        s["Stage"] = st; rank_tbl.append(s)
        sel[st] = s.sort_values("rank").iloc[0]["Method"]
    pd.concat(rank_tbl).round(4).to_csv(os.path.join(out, "enhancement_ranking.csv"), index=False)
    json.dump(sel, open(os.path.join(out, "selected_enhancement.json"), "w"), indent=2)
    print("SELECTED:", sel)

    # ---- bar charts
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for k, (st, col, title) in enumerate([("color", "UIQM", "Colour correction - UIQM"), ("dehaze", "UIQM", "Dehazing - UIQM"),
                                          ("denoise", "NoiseSigma", "Denoising - residual noise (lower=better)"), ("contrast", "UCIQE", "Contrast - UCIQE")]):
        s = summ[summ.Stage == st]; names = list(s.Method); vals = list(s[col])
        if st != "denoise": names = ["original"] + names; vals = [float(base[col])] + vals
        a_ = ax.flat[k]; bars = a_.bar(names, vals, color=["#9aa5b1"] + ["#2b7bba"] * (len(names) - 1) if st != "denoise" else "#2b7bba")
        a_.set_title(title, fontsize=11, fontweight="bold"); a_.tick_params(axis="x", rotation=20)
        for b_, v in zip(bars, vals): a_.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("AquaInspect - enhancement method comparison (mean over %d images)" % len(files), fontweight="bold")
    plt.tight_layout(); fig.savefig(os.path.join(out, "enhancement_comparison.png"), dpi=140); plt.close(fig)

    # ---- sample grid on one image
    img = E.resize_max(cv2.imread(files[2]), 500)
    panels = [("original", img)] + [(f"{st}:{m}", E.apply_stage(img, st, m)) for st, ms in STAGES.items() for m in ms if m != "none"]
    n = len(panels); cols = 4; rows_ = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows_, cols, figsize=(14, 3.3 * rows_))
    for a_ in ax.flat: a_.axis("off")
    for a_, (t, im) in zip(ax.flat, panels): a_.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); a_.set_title(t, fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out, "enhancement_samples.png"), dpi=120); plt.close(fig)

if __name__ == "__main__":
    main()
