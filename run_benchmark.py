"""
Synthetic-benchmark evaluation: (1) full-reference enhancement metrics, (2) downstream-validated
stage-wise method selection, (3) ablation study, (4) crack-detector method comparison, (5) severity agreement.

Usage: python scripts/run_benchmark.py [--n 24]
Outputs in results/: benchmark_enhancement_fullref.csv, benchmark_stage_selection.csv, selected_config.json,
                     ablation_study.csv/.png, detector_comparison.csv/.png, benchmark_examples.png
"""
import argparse, json, os, sys, time, warnings
import cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aquainspect import enhancement as E, metrics as M
from aquainspect.pipeline import run_inspection
from aquainspect.severity import severity
from aquainspect.synthetic import make_benchmark
from aquainspect.crack_detector import CrackDetector
from aquainspect.crack_refine import refine_crack_mask
from aquainspect import corrosion as C

TOL = 3  # px boundary tolerance for thin cracks


def gt_level(s):
    L = float(skeletonize(s["crack_gt"] > 0).sum()); pct = 100.0 * (s["rust_gt"] > 0).mean()
    return severity(pct, L, s["clean"].shape)["severity_level"]


def eval_config(bench, config, enhance=True, refine=True, label=""):
    rows = []
    for s in bench:
        r = run_inspection(s["degraded"], enhance_config=config, enhance=enhance, refine=refine, with_map=False)
        ck = M.seg_metrics(r["crack_mask"], s["crack_gt"], tol=TOL)
        ci = M.seg_metrics(r["crack_mask"], s["crack_gt"], tol=0)
        co = M.seg_metrics(r["corrosion_mask"], s["rust_gt"], tol=0)
        rows.append({"crack_F1": ck["F1"], "crack_P": ck["Precision"], "crack_R": ck["Recall"], "crack_IoU": ci["IoU"],
                     "corr_F1": co["F1"], "corr_P": co["Precision"], "corr_R": co["Recall"], "corr_IoU": co["IoU"],
                     "PSNR": M.psnr(s["clean"], r["enhanced"]), "SSIM": M.ssim(s["clean"], r["enhanced"]),
                     "UIQM": M.uiqm(r["enhanced"]), "UCIQE": M.uciqe(r["enhanced"]),
                     "level_ok": float(r["severity"]["severity_level"] == gt_level(s))})
    d = pd.DataFrame(rows).mean().to_dict(); d["objective"] = (d["crack_F1"] + d["corr_F1"]) / 2; d["Config"] = label
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24); a = ap.parse_args()
    out = os.path.join(ROOT, "results"); os.makedirs(out, exist_ok=True)
    bench = make_benchmark(a.n); t0 = time.time()
    print(f"benchmark: {a.n} synthetic images", flush=True)

    # ---- (1) full-reference enhancement metrics for each single method
    fr = []
    for st, ms in (("color", E.COLOR_METHODS), ("dehaze", E.DEHAZE_METHODS), ("denoise", E.DENOISE_METHODS), ("contrast", E.CONTRAST_METHODS)):
        for m in ms:
            vals = []
            for s in bench:
                o = E.apply_stage(s["degraded"], st, m)
                vals.append((M.psnr(s["clean"], o), M.ssim(s["clean"], o), M.uiqm(o), M.uciqe(o)))
            v = np.mean(vals, 0); fr.append({"Stage": st, "Method": m, "PSNR_vs_clean": v[0], "SSIM_vs_clean": v[1], "UIQM": v[2], "UCIQE": v[3]})
    full_ref = pd.DataFrame(fr).round(4); full_ref.to_csv(os.path.join(out, "benchmark_enhancement_fullref.csv"), index=False)
    print(full_ref.to_string(), flush=True)

    # ---- (2) greedy stage-wise selection by downstream detection score
    cfg = {"color": "none", "dehaze": "none", "denoise": "none", "contrast": "none"}
    sel_rows = []
    raw = eval_config(bench, None, enhance=False, refine=True, label="no enhancement"); sel_rows.append({**raw, "Stage": "baseline", "Method": "raw"})
    print("raw objective", round(raw["objective"], 3), flush=True)
    for st, ms in (("color", E.COLOR_METHODS), ("dehaze", E.DEHAZE_METHODS), ("denoise", E.DENOISE_METHODS), ("contrast", E.CONTRAST_METHODS)):
        cands = []
        for m in ms:
            c = {**cfg, st: m}; res = eval_config(bench, c, True, True, f"{st}={m}"); res.update(Stage=st, Method=m); cands.append(res); sel_rows.append(res)
            print(st, m, round(res["objective"], 3), flush=True)
        best = max(cands, key=lambda r: r["objective"]); none = next(r for r in cands if r["Method"] == "none")
        cfg[st] = "none" if best["objective"] - none["objective"] < 0.01 else best["Method"]   # parsimony
        print("->", st, "=", cfg[st], flush=True)
    pd.DataFrame(sel_rows).round(4).to_csv(os.path.join(out, "benchmark_stage_selection.csv"), index=False)
    json.dump(cfg, open(os.path.join(out, "selected_config.json"), "w"), indent=2)

    # ---- (3) ablation
    S = ["color", "dehaze", "denoise", "contrast"]
    abl = [eval_config(bench, None, False, False, "A0  raw input, no refinement"),
           eval_config(bench, None, False, True, "A1  raw + crack refinement")]
    acc = {s: "none" for s in S}
    for k, s in enumerate(S):
        acc[s] = cfg[s]; abl.append(eval_config(bench, dict(acc), True, True, f"A{k+2}  + {s} ({cfg[s]})"))
    abl.append(eval_config(bench, cfg, True, False, "Full chain, NO crack refinement"))
    for s in S:
        c = {**cfg, s: "none"}; abl.append(eval_config(bench, c, True, True, f"Full minus {s}"))
    abl.append(eval_config(bench, cfg, True, True, "FULL AquaInspect"))
    ab = pd.DataFrame(abl).round(4); ab.to_csv(os.path.join(out, "ablation_study.csv"), index=False)
    print(ab[["Config", "crack_F1", "corr_F1", "objective", "PSNR", "SSIM", "level_ok"]].to_string(), flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    cum = ab.iloc[:6]
    x = np.arange(len(cum)); w = 0.38
    ax[0].bar(x - w / 2, cum.crack_F1, w, label="Crack F1", color="#2b7bba"); ax[0].bar(x + w / 2, cum.corr_F1, w, label="Corrosion F1", color="#d9822b")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["raw", "raw+refine", "+color", "+dehaze", "+denoise", "+contrast"], rotation=20); ax[0].set_ylim(0, 1); ax[0].legend(); ax[0].set_title("Cumulative ablation (adding stages)", fontweight="bold")
    loo = ab.iloc[7:]
    ax[1].barh(loo.Config, loo.objective, color=["#9aa5b1"] * (len(loo) - 1) + ["#2a9d5c"]); ax[1].set_xlim(0, 1); ax[1].set_title("Leave-one-out (objective = mean of crack & corrosion F1)", fontweight="bold")
    for i, v in enumerate(loo.objective): ax[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out, "ablation_study.png"), dpi=140); plt.close(fig)

    # ---- (4) crack detector comparison on the enhanced benchmark (M2's three modes, with/without refinement)
    dets = []
    for method in ("canny", "blackhat", "hybrid"):
        for ref in (False, True):
            ms, tt = [], 0.0
            for s in bench:
                en = E.enhance(s["degraded"], cfg); t = time.perf_counter()
                d = CrackDetector(); m_, _ = d.segment_cracks(d.preprocess(en)[1], method=method)
                if ref: m_, _ = refine_crack_mask(m_)
                tt += time.perf_counter() - t
                r = M.seg_metrics(m_, s["crack_gt"], tol=TOL); r["IoU"] = M.seg_metrics(m_, s["crack_gt"], 0)["IoU"]; ms.append(r)
            row = pd.DataFrame(ms).mean().to_dict(); row.update(Method=method, Refinement="yes" if ref else "no", Latency_ms=1000 * tt / len(bench)); dets.append(row)
    dd = pd.DataFrame(dets).round(4)[["Method", "Refinement", "IoU", "Precision", "Recall", "F1", "Latency_ms"]]; dd.to_csv(os.path.join(out, "detector_comparison.csv"), index=False); print(dd.to_string(), flush=True)
    fig, a_ = plt.subplots(figsize=(9, 4.8)); lab = [f"{m}\n{'+refine' if r == 'yes' else 'raw'}" for m, r in zip(dd.Method, dd.Refinement)]
    x = np.arange(len(dd)); a_.bar(x - 0.27, dd.Precision, 0.27, label="Precision"); a_.bar(x, dd.Recall, 0.27, label="Recall"); a_.bar(x + 0.27, dd.F1, 0.27, label="F1")
    a_.set_xticks(x); a_.set_xticklabels(lab); a_.set_ylim(0, 1); a_.legend(); a_.set_title("Crack detector comparison (synthetic GT, %d px tolerance)" % TOL, fontweight="bold")
    plt.tight_layout(); fig.savefig(os.path.join(out, "detector_comparison.png"), dpi=140); plt.close(fig)

    # ---- example figure
    rows = []
    for s in bench[:4]:
        r = run_inspection(s["degraded"], enhance_config=cfg, with_map=True)
        rows.append(np.hstack([s["clean"], s["degraded"], r["enhanced"], r["crack_overlay"], r["damage_map"]]))
    cv2.imwrite(os.path.join(out, "benchmark_examples.jpg"), cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("TOTAL", round(time.time() - t0), "s  SELECTED", cfg)

if __name__ == "__main__":
    main()
