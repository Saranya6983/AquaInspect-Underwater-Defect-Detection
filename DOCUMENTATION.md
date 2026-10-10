# AquaInspect - System Documentation

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
2. **Member 3's severity formula could not work as integrated.** `0.6*corrosion% + 0.4*crack_length` used a raw pixel crack length (hard-coded to `10` in the batch script). Result on Member 3's own dataset: **all 39 images "Low"** (max score 12.9). Crack length in pixels is also resolution-dependent. The crack term is now normalised by image size and both terms are scaled to 0-100 (see §5). On the same images the levels are now: {'Low': 23, 'Medium': 15, 'High': 1}.
3. **Member 2's detector over-detects on real scenes.** It reports tens of "crack segments" on gravel, marine growth, divers, ROVs and ripples (29-105 segments per image on the first eight sample images). `crack_refine.py` post-filters its mask using length, thinness, a *mesh test* (skeleton junctions per 100 px: cracks 0-5, texture 7-28 as measured) and isolation from other detections. On the synthetic benchmark this raises crack precision from 0.73 to 0.92 and F1 from 0.71 to 0.78.
4. **Two files in Member 3's dataset were empty (0 bytes)** (`222.jpeg`, `00.jpeg`) and were dropped (39 of 41 images remain).
5. Images larger than 1280 px on the long side are down-scaled before analysis (speed; NLM and skeletonisation are slow on 4K frames). Measurements are therefore in analysed-image pixels.

## 3. Enhancement methodology and comparison

Four stages are compared, each against the original image. Two complementary comparisons are reported because they answer different questions.

### 3.1 Real images (n = 20) - no-reference quality
UIQM and UCIQE are no-reference underwater-quality measures (higher = better). "PSNR/SSIM vs input" only says how strongly a method *changes* the image - there is no clean reference for real underwater photos. UIQM here uses a simplified UIConM term, so use it for *relative* comparison only.

| Stage    | Method      |   UIQM |   UCIQE |   NoiseSigma | PSNR_vs_input   |   SSIM_vs_input |   Time_s |
|:---------|:------------|-------:|--------:|-------------:|:----------------|----------------:|---------:|
| baseline | original    |  2.151 |   0.389 |        1.632 | -               |           1     |    0     |
| color    | gray_world  |  2.139 |   0.369 |        1.758 | 21.518          |           0.924 |    0.011 |
| color    | white_patch |  2.194 |   0.407 |        2.03  | 22.952          |           0.946 |    0.011 |
| color    | red_comp    |  2.16  |   0.35  |        1.713 | 21.746          |           0.908 |    0.018 |
| dehaze   | dcp         |  2.267 |   0.425 |        1.864 | 22.911          |           0.903 |    0.037 |
| dehaze   | udcp        |  2.223 |   0.466 |        2.14  | 16.524          |           0.696 |    0.036 |
| denoise  | gaussian    |  1.818 |   0.382 |        0.307 | 34.278          |           0.923 |    0.002 |
| denoise  | median      |  1.534 |   0.384 |        0.564 | 32.177          |           0.867 |    0.002 |
| denoise  | bilateral   |  1.696 |   0.385 |        0.993 | 36.822          |           0.923 |    0.038 |
| denoise  | nlm         |  1.207 |   0.384 |        0.613 | 32.58           |           0.839 |    0.723 |
| contrast | he          |  2.154 |   0.415 |        3.018 | 14.431          |           0.694 |    0.002 |
| contrast | clahe       |  2.532 |   0.402 |        3.257 | 21.008          |           0.798 |    0.007 |

Ranking these metrics alone selects `{'color': 'white_patch', 'dehaze': 'dcp', 'denoise': 'gaussian', 'contrast': 'he'}`. **This was not adopted**, because the metrics reward things that hurt detection: e.g. Gaussian blur wins "noise" by blurring thin cracks, and no-reference scores know nothing about the downstream task.

### 3.2 Synthetic benchmark (n = 24) - full-reference
Because the benchmark has a clean version of every image, true PSNR/SSIM can be computed for each single-stage method applied to the degraded image:

| Stage    | Method      |   PSNR_vs_clean |   SSIM_vs_clean |   UIQM |   UCIQE |
|:---------|:------------|----------------:|----------------:|-------:|--------:|
| color    | none        |          11.281 |           0.394 |  2.741 |   0.31  |
| color    | gray_world  |          11.992 |           0.389 |  2.788 |   0.239 |
| color    | white_patch |          11.709 |           0.186 |  2.968 |   0.391 |
| color    | red_comp    |          12.206 |           0.406 |  3.044 |   0.228 |
| dehaze   | none        |          11.281 |           0.394 |  2.741 |   0.31  |
| dehaze   | dcp         |          11.04  |           0.376 |  2.722 |   0.315 |
| dehaze   | udcp        |           9.835 |           0.27  |  2.526 |   0.354 |
| denoise  | none        |          11.281 |           0.394 |  2.741 |   0.31  |
| denoise  | gaussian    |          11.312 |           0.564 |  1.768 |   0.294 |
| denoise  | median      |          11.302 |           0.561 |  1.219 |   0.294 |
| denoise  | bilateral   |          11.305 |           0.563 |  0.936 |   0.292 |
| denoise  | nlm         |          11.239 |           0.541 |  0.571 |   0.291 |
| contrast | none        |          11.281 |           0.394 |  2.741 |   0.31  |
| contrast | he          |          10.897 |           0.17  |  2.716 |   0.415 |
| contrast | clahe       |          13.989 |           0.309 |  3.104 |   0.33  |

### 3.3 Selected chain (downstream-validated)
Methods were chosen **stage by stage by the detection score they produce** (mean of crack F1 and corrosion F1; "none" is preferred unless a method gains ≥ 0.01):

`{'color': 'red_comp', 'dehaze': 'none', 'denoise': 'gaussian', 'contrast': 'clahe'}`

This agrees with Member 1's choice of CLAHE and shows that dehazing (DCP/UDCP) did not help for this degradation model. The selection was made on synthetic data - **re-run `scripts/run_benchmark.py` / use your annotated images before claiming it for real footage**; any stage can be changed in the app sidebar.

## 4. Crack and corrosion methodology
**Cracks (Member 2):** grayscale → bilateral filter → black-hat (dark thin structures) → adaptive threshold → 3×3 closing → connected-component filtering → skeletonisation (scikit-image) → length = skeleton pixels, width = 2 × distance transform on the skeleton, area = mask pixels. Optional mm/px calibration converts to mm.
Detector comparison (synthetic GT, 3 px tolerance for thin lines; IoU is exact-pixel):

| Method   | Refinement   |   IoU |   Precision |   Recall |    F1 |   Latency_ms |
|:---------|:-------------|------:|------------:|---------:|------:|-------------:|
| canny    | no           | 0.002 |       0.178 |    0.13  | 0.139 |       17.959 |
| canny    | yes          | 0.002 |       0.178 |    0.13  | 0.139 |       20.282 |
| blackhat | no           | 0.181 |       0.344 |    0.933 | 0.444 |       18.871 |
| blackhat | yes          | 0.273 |       0.636 |    0.596 | 0.586 |       33.309 |
| hybrid   | no           | 0.423 |       0.729 |    0.772 | 0.712 |       18.834 |
| hybrid   | yes          | 0.509 |       0.917 |    0.713 | 0.776 |       25.636 |

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

| Config                          |   crack_F1 |   crack_P |   crack_R |   crack_IoU |   corr_F1 |   corr_P |   corr_R |   corr_IoU |   PSNR |   SSIM |   severity_level_acc |
|:--------------------------------|-----------:|----------:|----------:|------------:|----------:|---------:|---------:|-----------:|-------:|-------:|---------------------:|
| A0  raw input, no refinement    |      0.441 |     0.619 |     0.362 |       0.28  |     0.125 |    0.125 |    0.125 |      0.125 | 11.281 |  0.394 |                0.125 |
| A1  raw + crack refinement      |      0.425 |     0.625 |     0.347 |       0.271 |     0.125 |    0.125 |    0.125 |      0.125 | 11.281 |  0.394 |                0.125 |
| A2  + color (red_comp)          |      0.508 |     0.708 |     0.422 |       0.332 |     0.153 |    0.708 |    0.14  |      0.14  | 12.206 |  0.406 |                0.125 |
| A3  + dehaze (none)             |      0.508 |     0.708 |     0.422 |       0.332 |     0.153 |    0.708 |    0.14  |      0.14  | 12.206 |  0.406 |                0.125 |
| A4  + denoise (gaussian)        |      0.422 |     0.667 |     0.345 |       0.268 |     0.407 |    0.912 |    0.313 |      0.313 | 12.299 |  0.608 |                0.167 |
| A5  + contrast (clahe)          |      0.776 |     0.917 |     0.713 |       0.509 |     0.516 |    1     |    0.401 |      0.401 | 14.929 |  0.666 |                0.417 |
| Full chain, NO crack refinement |      0.712 |     0.729 |     0.772 |       0.423 |     0.516 |    1     |    0.401 |      0.401 | 14.929 |  0.666 |                0.583 |
| Full minus color                |      0.771 |     0.898 |     0.704 |       0.511 |     0.125 |    0.125 |    0.125 |      0.125 | 13.528 |  0.654 |                0.167 |
| Full minus dehaze               |      0.776 |     0.917 |     0.713 |       0.509 |     0.516 |    1     |    0.401 |      0.401 | 14.929 |  0.666 |                0.417 |
| Full minus denoise              |      0.469 |     0.649 |     0.409 |       0.293 |     0.143 |    0.708 |    0.135 |      0.135 | 15.358 |  0.274 |                0.125 |
| Full minus contrast             |      0.422 |     0.667 |     0.345 |       0.268 |     0.407 |    0.912 |    0.313 |      0.313 | 12.299 |  0.608 |                0.167 |
| FULL AquaInspect                |      0.776 |     0.917 |     0.713 |       0.509 |     0.516 |    1     |    0.401 |      0.401 | 14.929 |  0.666 |                0.417 |

Reading it: the full chain raises the mean of crack and corrosion F1 from 0.28 (no enhancement) to 0.65. Removing **denoising** hurts most (noise becomes false cracks), then **colour correction** (corrosion recall collapses without it - red is absorbed first and the rust hue disappears), then **contrast (CLAHE)**; **dehazing** contributes nothing here. The crack **refinement** filter trades a little recall for much higher precision. Severity-level agreement is 42% - classification is the weakest output and should be treated as indicative.

![ablation](../results/ablation_study.png)

### 6.3 Real-image batch run (`results/batch/`)
20 sample images: {'Medium': 10, 'Low': 9, 'High': 1}. Mean processing time 0.3 s/image. These are *outputs*, not accuracy figures.

## 7. Limitations (classical methods)
* **Marine growth / biofouling** - dark texture looks like cracks; green growth is suppressed for corrosion, brown/orange growth and rust-coloured sediment are not.
* **Shadows and light beams** - deep shadow is removed from the corrosion mask, but shadow *edges* and torch-beam edges create crack-like lines.
* **Clutter** - divers, ROVs, cables, surface ripples produce edges; the refinement filter removes most (e.g. sample image 01: 29 raw segments → 1) but residual false detections remain, visible in the app's crack tab.
* **Corrosion recall** - the fixed HSV thresholds (S ≥ 100, V ≥ 50) miss rust in dim water: on the benchmark precision is 1.00 but recall only 0.40.
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
