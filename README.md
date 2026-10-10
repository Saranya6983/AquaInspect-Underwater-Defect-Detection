# 🌊 AquaInspect

Classical image-processing system for **defect detection and damage assessment of underwater pipelines and bridge piers**:
enhancement → crack detection & measurement → corrosion detection → severity score → colour-coded damage map → report.

## Quick start
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open the URL Streamlit prints. Upload an image or pick one of the 20 bundled samples.

| Page | What it does |
|---|---|
| 🔍 Inspect | Original vs enhanced, crack mask/skeleton/measurements, corrosion mask/area, damage map, live enhancement-method comparison, **PDF / CSV / PNG report download** |
| 📊 Batch dashboard | Run many images; severity distribution, scatter, ranked table, damage-map gallery, CSV export |
| 🧪 Evaluation | Enhancement comparison, synthetic-benchmark ablation, detector comparison, **evaluation on your annotated masks** |
| ℹ️ About | Pipeline, module ownership, limitations |

Sidebar → *Pipeline settings*: choose the method for each enhancement stage, toggle the crack refinement filter, set mm/pixel calibration.

## Folder layout
```
app.py                      Streamlit app (Member 4)
aquainspect/                Python package
  enhancement.py            Member 1  - colour correction, dehazing, denoising, contrast
  crack_detector.py         Member 2  - UNMODIFIED
  corrosion.py              Member 3  - UNMODIFIED
  severity.py               severity score + damage map (Member 3's formula, size-normalised)
  crack_refine.py           false-positive filter for crack masks
  pipeline.py               integrated pipeline + input validation
  metrics.py  synthetic.py  evaluate_gt.py  report.py
scripts/                    run_enhancement_eval, run_benchmark, run_batch, make_docs
tests/test_pipeline.py      7 tests   (python tests/test_pipeline.py)
data/sample_images/         20 underwater images (Member 1)   data/corrosion_samples/  39 images (Member 3)
data/ground_truth/          ← put manual annotations here (see below)
results/                    all CSVs / figures produced by the scripts
docs/DOCUMENTATION.md       full methodology, evaluation, limitations
member_work/                the three members' original submissions, untouched
```

## Use as a library
```python
from aquainspect.pipeline import run_inspection
r = run_inspection("pier.jpg")            # path, bytes or ndarray
r["severity"]["severity_level"], r["crack_metrics"]["crack_length_px"], r["corrosion_metrics"]["corrosion_percent"]
r["damage_map"]                           # BGR image
```

## ⚠️ Read before submitting
1. **Real-image accuracy is not yet measured.** The abstract requires IoU / precision / recall / F1 against *manually annotated* ground truth; none was supplied. Annotate 15-20 images (white = defect) as `data/ground_truth/<image_stem>_crack.png` and `<image_stem>_corrosion.png`, then use *Evaluation → Detection vs annotated ground truth*.
2. The ablation and method selection in `results/` come from a **synthetic benchmark** (exact masks, simulated underwater degradation). It validates the pipeline; it is not real-world accuracy.
3. Integration fixes to Members 1-3's work (severity formula, crack false positives, empty dataset files) are listed in `docs/DOCUMENTATION.md` §2.
