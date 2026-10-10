"""Run with:  python -m pytest -q   (or: python tests/test_pipeline.py)"""
import os, sys, warnings
import cv2, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aquainspect.pipeline import run_inspection, load_image, InvalidImageError
from aquainspect.severity import severity, classify, damage_map
from aquainspect.metrics import seg_metrics
from aquainspect.synthetic import make_sample
from aquainspect.report import build_pdf
from aquainspect import enhancement as E


def test_invalid_inputs():
    for bad in (b"not an image", np.zeros((10, 10, 3), np.uint8), "/nonexistent.jpg"):
        try:
            load_image(bad); assert False, "should raise"
        except InvalidImageError:
            pass


def test_gray_and_alpha_inputs_are_converted():
    assert load_image(np.full((100, 100), 128, np.uint8)).shape == (100, 100, 3)
    assert load_image(np.full((100, 100, 4), 128, np.uint8)).shape == (100, 100, 3)


def test_every_enhancement_method_runs_and_keeps_shape():
    img = make_sample(1)["degraded"]
    for stage, methods in (("color", E.COLOR_METHODS), ("dehaze", E.DEHAZE_METHODS), ("denoise", E.DENOISE_METHODS), ("contrast", E.CONTRAST_METHODS)):
        for m in methods:
            o = E.apply_stage(img, stage, m); assert o.shape == img.shape and o.dtype == np.uint8


def test_severity_monotonic_and_levels():
    shape = (500, 500)
    a, b = severity(0, 0, shape), severity(20, 300, shape)
    assert a["severity_level"] == "Low" and b["severity_score"] > a["severity_score"]
    assert severity(100, 10_000, shape)["severity_score"] <= 100
    assert [classify(x) for x in (0, 19.9, 20, 49.9, 50)] == ["Low", "Low", "Medium", "Medium", "High"]
    # resolution independence: same relative crack -> same score
    assert abs(severity(5, 100, (200, 200))["severity_score"] - severity(5, 200, (400, 400))["severity_score"]) < 0.1


def test_seg_metrics():
    g = np.zeros((50, 50), np.uint8); g[10:20, 10:20] = 255
    assert seg_metrics(g, g)["F1"] == 1.0 and seg_metrics(np.zeros_like(g), g)["Recall"] == 0.0


def test_full_pipeline_on_synthetic_finds_rust_and_cracks():
    s = make_sample(102)  # concrete with rust
    r = run_inspection(s["clean"], enhance=False)
    assert r["crack_metrics"]["crack_detected"]
    assert r["severity"]["severity_level"] in ("Low", "Medium", "High")
    assert r["damage_map"].shape == r["enhanced"].shape
    pdf = build_pdf("t.png", r, {"color": "none"}); assert pdf[:4] == b"%PDF"


def test_clean_blank_image_is_low_severity():
    r = run_inspection(np.full((200, 300, 3), 140, np.uint8), enhance=False)
    assert r["severity"]["severity_level"] == "Low" and r["crack_metrics"]["crack_length_px"] == 0


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
