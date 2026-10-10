"""Regenerates docs/DOCUMENTATION.md from results/*.csv (so numbers always match the results)."""
import os, json, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: pd.read_csv(os.path.join(ROOT, "results", f))
md = lambda df: df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)
try:
    import tabulate  # noqa
except Exception:
    md = lambda df: "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.values)

enh = R("enhancement_summary.csv").round(3).fillna("-")
fr = R("benchmark_enhancement_fullref.csv").round(3)
ab = R("ablation_study.csv"); det = R("detector_comparison.csv").round(3)
cfg = json.load(open(os.path.join(ROOT, "results", "selected_config.json")))
metric_sel = json.load(open(os.path.join(ROOT, "results", "selected_enhancement.json")))
ab_t = ab[["Config", "crack_F1", "crack_P", "crack_R", "crack_IoU", "corr_F1", "corr_P", "corr_R", "corr_IoU", "PSNR", "SSIM", "level_ok"]].round(3)
ab_t = ab_t.rename(columns={"level_ok": "severity_level_acc"})
full = ab[ab.Config == "FULL AquaInspect"].iloc[0]; raw = ab[ab.Config.str.startswith("A0")].iloc[0]
b_old = R("batch_corrosion_samples/batch_results.csv")
b_s = R("batch/batch_results.csv")

doc = f"""# AquaInspect - System Documentation

*Classical image-processing defect detection and severity assessment of underwater pipelines and bridge piers.*

## 1. System overview

```
 image ──► validate ──► ENHANCE (Member 1) ──► CRACKS (Member 2) ──► refine ──┐
                         colour → dehaze →       grayscale → bilateral →      │
                         denoise → contrast      black-hat → adaptive thr →   ├─► SEVERITY ─► DAMAGE MAP ─► REPORT
                                                 morphology → skeleton →      │   (Member 3)    (colour-coded)   (PDF/CSV)
                         CORROSION (Member 3) ─────────────────────────────── ┘
                         HSV → morphology → remove green growth & shadows → area %
```

| Layer | File | Owner |
|---|---|---|
| Enhancement | `aquainspect/enhancement.py` (original script kept in `member_work/member1_enhancement/`) | Member 1 (refactored into a multi-method module by Member 4) |
| Crack detection & measurement | `aquainspect/crack_detector.py` - **unmodified** copy of Member 2's module | Member 2 |
| Corrosion detection | `aquainspect/corrosion.py` - **unmodified** copy of Member 3's script | Member 3 |
| Severity + damage map | `aquainspect/severity.py` | Member 3's formula, normalised by Member 4 |
| Crack refinement filter | `aquainspect/crack_refine.py` | Member 4 |
| Pipeline, validation, errors | `aquainspect/pipeline.py` | Member 4 |
| Metrics (UIQM, UCIQE, PSNR, SSIM, IoU, P/R/F1) | `aquainspect/metrics.py` | Member 4 |
| Synthetic benchmark | `aquainspect/synthetic.py` | Member 4 |
| Reports | `aquainspect/report.py` | Member 4 |
| Streamlit app | `app.py` | Member 4 |
| Evaluation / batch scripts | `scripts/` | Member 4 |

Everything the members originally submitted is preserved untouched in `member_work/`.

## 2. Integration changes made to the members' work (please read)

Integrating the three modules exposed problems that had to be fixed at the integration layer. These are the decisions the team should know about and may want to reflect in the paper:

1. **Member 1 had one fixed chain, but the project requires comparing methods.** The original script (LAB-CLAHE → NLM denoise) is preserved as `MEMBER1_CONFIG`; `enhancement.py` adds gray-world / white-patch / red-channel colour correction, DCP / underwater-DCP dehazing, Gaussian / median / bilateral denoising and HE / CLAHE. (The original script also had a missing comma that silently skipped two input images - the new module reads whatever is in the folder.)
2. **Member 3's severity formula could not work as integrated.** `0.6*corrosion% + 0.4*crack_length` used a raw pixel crack length (hard-coded to `10` in the batch script). Result on Member 3's own dataset: **all {len(b_old)} images "Low"** (max score 12.9). Crack length in pixels is also resolution-dependent. The crack term is now normalised by image size and both terms are scaled to 0-100 (see §5). On the same images the levels are now: {b_old.Severity_Level.value_counts().to_dict()}.
3. **Member 2's detector over-detects on real scenes.** It reports tens of "crack segments" on gravel, marine growth, divers, ROVs and ripples (29-105 segments per image on the first eight sample images). `crack_refine.py` post-filters its mask using length, thinness, a *mesh test* (skeleton junctions per 100 px: cracks 0-5, texture 7-28 as measured) and isolation from other detections. On the synthetic benchmark this raises crack precision from {ab[ab.Config=='Full chain, NO crack refinement'].crack_P.iloc[0]:.2f} to {full.crack_P:.2f} and F1 from {ab[ab.Config=='Full chain, NO crack refinement'].crack_F1.iloc[0]:.2f} to {full.crack_F1:.2f}.
4. **Two files in Member 3's dataset were empty (0 bytes)** (`222.jpeg`, `00.jpeg`) and were dropped (39 of 41 images remain).
5. Images larger than 1280 px on the long side are down-scaled before analysis (speed; NLM and skeletonisation are slow on 4K frames). Measurements are therefore in analysed-image pixels.

## 3. Enhancement methodology and comparison

Four stages are compared, each against the original image. Two complementary comparisons are reported because they answer different questions.

### 3.1 Real images (n = {len(R('enhancement_per_image.csv').Image.unique())}) - no-reference quality
UIQM and UCIQE are no-reference underwater-quality measures (higher = better). "PSNR/SSIM vs input" only says how strongly a method *changes* the image - there is no clean reference for real underwater photos. UIQM here uses a simplified UIConM term, so use it for *relative* comparison only.

{md(enh)}

Ranking these metrics alone selects `{metric_sel}`. **This was not adopted**, because the metrics reward things that hurt detection: e.g. Gaussian blur wins "noise" by blurring thin cracks, and no-reference scores know nothing about the downstream task.

### 3.2 Synthetic benchmark (n = 24) - full-reference
Because the benchmark has a clean version of every image, true PSNR/SSIM can be computed for each single-stage method applied to the degraded image:

{md(fr)}

### 3.3 Selected chain (downstream-validated)
Methods were chosen **stage by stage by the detection score they produce** (mean of crack F1 and corrosion F1; "none" is preferred unless a method gains ≥ 0.01):

`{cfg}`

This agrees with Member 1's choice of CLAHE and shows that dehazing (DCP/UDCP) did not help for this degradation model. The selection was made on synthetic data - **re-run `scripts/run_benchmark.py` / use your annotated images before claiming it for real footage**; any stage can be changed in the app sidebar.

## 4. Crack and corrosion methodology
**Cracks (Member 2):** grayscale → bilateral filter → black-hat (dark thin structures) → adaptive threshold → 3×3 closing → connected-component filtering → skeletonisation (scikit-image) → length = skeleton pixels, width = 2 × distance transform on the skeleton, area = mask pixels. Optional mm/px calibration converts to mm.
Detector comparison (synthetic GT, 3 px tolerance for thin lines; IoU is exact-pixel):

{md(det)}

The hybrid mode (black-hat + adaptive threshold) is best, consistent with Member 2's choice; Canny alone fails after noise.

**Corrosion (Member 3):** BGR→HSV, rust range H 5-25 / S ≥ 100 / V ≥ 50, 5×5 open+close, then green (marine growth, H 35-85) and dark (V < 50, shadows) pixels are removed; area % = rust pixels / image pixels.

## 5. Severity scoring and damage map
```
crack_density = crack_length_px / sqrt(W*H)          crack_index     = min(100, 100 * crack_density / 2)
corrosion_idx = min(100, 100 * corrosion% / 40)
score = 0.6 * corrosion_idx + 0.4 * crack_index      Low < 20 <= Medium < 50 <= High
```
Weights and cut-offs are Member 3's; saturation points (40 % rust, 2 image-lengths of crack) are Member 4's engineering assumptions and are **not calibrated against an inspection standard**.
The damage map divides the image into a ~24-cell grid and classifies each cell from its local crack and corrosion density (green Low, amber Medium, red High; undamaged cells untinted).

## 6. Evaluation

### 6.1 What was and was not measured
* **Real-image detection accuracy (IoU/precision/recall/F1 on real photos) has NOT been measured** - no manually annotated ground truth was supplied. The abstract requires it. `aquainspect/evaluate_gt.py` and the *Evaluation → Detection vs annotated ground truth* tab compute it as soon as masks are placed in `data/ground_truth/` (`<image>_crack.png`, `<image>_corrosion.png`). **Annotating 15-20 images is the main remaining task for the team.**
* What *was* measured is a **synthetic benchmark** with exact masks (24 procedural concrete/steel images, cracks, rust, marine-growth and shadow distractors, simulated underwater attenuation/haze/noise). It validates pipeline logic and supports the ablation but is a **proxy**: scores here do not transfer to real accuracy. The benchmark's rust colour lies inside the HSV range of Member 3's rule by construction, so corrosion scores here are optimistic for colour matching (and still show low recall under dim lighting).

### 6.2 Ablation study (synthetic, n = 24)
Cumulative additions then leave-one-out. `severity_level_acc` = fraction of images whose Low/Medium/High matches the level computed from the exact ground-truth masks.

{md(ab_t)}

Reading it: the full chain raises the mean of crack and corrosion F1 from {raw.objective:.2f} (no enhancement) to {full.objective:.2f}. Removing **denoising** hurts most (noise becomes false cracks), then **colour correction** (corrosion recall collapses without it - red is absorbed first and the rust hue disappears), then **contrast (CLAHE)**; **dehazing** contributes nothing here. The crack **refinement** filter trades a little recall for much higher precision. Severity-level agreement is {full.level_ok:.0%} - classification is the weakest output and should be treated as indicative.

![ablation](../results/ablation_study.png)

### 6.3 Real-image batch run (`results/batch/`)
{len(b_s)} sample images: {b_s.Severity_Level.value_counts().to_dict()}. Mean processing time {b_s.Time_s.mean():.1f} s/image. These are *outputs*, not accuracy figures.

## 7. Limitations (classical methods)
* **Marine growth / biofouling** - dark texture looks like cracks; green growth is suppressed for corrosion, brown/orange growth and rust-coloured sediment are not.
* **Shadows and light beams** - deep shadow is removed from the corrosion mask, but shadow *edges* and torch-beam edges create crack-like lines.
* **Clutter** - divers, ROVs, cables, surface ripples produce edges; the refinement filter removes most (e.g. sample image 01: 29 raw segments → 1) but residual false detections remain, visible in the app's crack tab.
* **Corrosion recall** - the fixed HSV thresholds (S ≥ 100, V ≥ 50) miss rust in dim water: on the benchmark precision is {full.corr_P:.2f} but recall only {full.corr_R:.2f}.
* **Fine cracks** (< 2 px) can vanish after denoising or low resolution; width accuracy is ± 1 px.
* **Pixel units** unless calibrated; no perspective correction.
* **Severity scale** is heuristic, not standards-based.

## 8. Testing
`python tests/test_pipeline.py` - 7 tests: invalid/empty/tiny/grayscale/alpha inputs, all 11 enhancement methods, severity monotonicity / level boundaries / resolution independence, segmentation metrics, end-to-end pipeline + PDF generation, blank-image sanity. The Streamlit app was exercised headlessly through all four pages.

## 9. Reproducing the results
```
python scripts/run_enhancement_eval.py    # real-image enhancement comparison  (~3 min)
python scripts/run_benchmark.py --n 24    # synthetic benchmark, selection, ablation, detector comparison (~4 min)
python scripts/run_batch.py               # batch inspection of data/sample_images -> results/batch/
python scripts/make_docs.py               # regenerate this document from the CSVs
```
"""
open(os.path.join(ROOT, "docs", "DOCUMENTATION.md"), "w", encoding="utf-8").write(doc)
print("written", len(doc), "chars")
