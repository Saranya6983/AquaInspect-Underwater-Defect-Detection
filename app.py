"""
AquaInspect - Streamlit application (Member 4: frontend + integration).
Run:  streamlit run app.py
"""
import glob
import json
import os
import warnings

from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import cv2
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))

from AquaInspect import enhancement as E
from AquaInspect import metrics as M
from AquaInspect.evaluate_gt import annotated_images, evaluate_folder
from AquaInspect.pipeline import InvalidImageError, load_image, run_inspection

st.set_page_config(page_title="AquaInspect", page_icon="🌊", layout="wide")

RESULTS = os.path.join(ROOT, "results")
SAMPLES = sorted(glob.glob(os.path.join(ROOT, "data", "sample_images", "*")))
LEVEL_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}


# ------------------------------------------------------------------ look & feel
import base64

def _bg_data_uri():
    p = os.path.join(ROOT, "assets", "background.png")
    if os.path.exists(p):
        return "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()
    return ""


def inject_css():
    bg = _bg_data_uri()
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Poppins', sans-serif; }}

/* ---- background image ---- */
.stApp {{
    background: linear-gradient(180deg, rgba(4,16,30,.55), rgba(4,16,30,.80)),
                url("{bg}") center / cover fixed no-repeat;
}}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}

/* ---- hero ---- */
.hero {{
    padding: 1.6rem 2rem; margin-bottom: 1.2rem; border-radius: 22px;
    background: linear-gradient(120deg, rgba(14,116,144,.55), rgba(8,47,73,.55));
    border: 1px solid rgba(125,211,252,.25);
    box-shadow: 0 10px 40px rgba(0,0,0,.35);
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
}}
.hero h1 {{ margin: 0; font-size: 2.1rem; font-weight: 700; color: #fff; letter-spacing: .3px; }}
.hero h1 span {{ background: linear-gradient(90deg,#67e8f9,#38bdf8); -webkit-background-clip: text;
                 -webkit-text-fill-color: transparent; }}
.hero p {{ margin: .35rem 0 0; color: #bae6fd; font-size: .98rem; }}
.chips {{ margin-top: .8rem; display: flex; gap: .5rem; flex-wrap: wrap; }}
.chip {{ padding: .22rem .75rem; border-radius: 999px; font-size: .76rem; color: #e0f7ff;
         background: rgba(103,232,249,.14); border: 1px solid rgba(103,232,249,.35); }}

/* ---- glass cards ---- */
[data-testid="stMetric"] {{
    background: rgba(8,36,60,.62); border: 1px solid rgba(125,211,252,.22);
    border-radius: 18px; padding: 1rem 1.1rem;
    box-shadow: 0 6px 24px rgba(0,0,0,.30);
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    transition: transform .15s ease, border-color .15s ease;
}}
[data-testid="stMetric"]:hover {{ transform: translateY(-3px); border-color: rgba(103,232,249,.7); }}
[data-testid="stMetricLabel"] p {{ color: #7dd3fc; font-size: .8rem; text-transform: uppercase; letter-spacing: .8px; }}
[data-testid="stMetricValue"] {{ color: #fff; font-weight: 600; }}

[data-testid="stTabs"] [data-baseweb="tab-panel"] {{
    background: rgba(6,28,48,.55); border: 1px solid rgba(125,211,252,.15);
    border-radius: 0 0 18px 18px; padding: 1.2rem 1.2rem .8rem;
    backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
}}
[data-baseweb="tab-list"] {{ gap: .3rem; }}
button[data-baseweb="tab"] {{
    background: rgba(8,36,60,.55); border-radius: 12px 12px 0 0; padding: .55rem 1rem; color: #bae6fd;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg,#0891b2,#0369a1); color: #fff;
}}
[data-baseweb="tab-highlight"] {{ background: transparent !important; }}

[data-testid="stImage"] img {{ border-radius: 14px; border: 1px solid rgba(125,211,252,.25);
                               box-shadow: 0 6px 22px rgba(0,0,0,.4); }}
[data-testid="stFileUploader"] section {{
    background: rgba(8,36,60,.55); border: 2px dashed rgba(103,232,249,.55); border-radius: 18px;
}}
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}

/* ---- buttons ---- */
.stButton > button, .stDownloadButton > button {{
    background: linear-gradient(135deg,#06b6d4,#0369a1); color: #fff; border: 0; border-radius: 12px;
    padding: .55rem 1.2rem; font-weight: 600; box-shadow: 0 6px 18px rgba(6,182,212,.35);
    transition: transform .12s ease, box-shadow .12s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-2px); box-shadow: 0 10px 26px rgba(6,182,212,.55); color: #fff;
}}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(5,24,42,.93), rgba(3,14,26,.96));
    border-right: 1px solid rgba(125,211,252,.18);
}}
[data-testid="stSidebar"] h1 {{ font-size: 1.5rem; color: #67e8f9; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    background: rgba(8,36,60,.6); border: 1px solid rgba(125,211,252,.18); border-radius: 12px;
    padding: .5rem .8rem; margin-bottom: .35rem; width: 100%; transition: all .15s ease;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ border-color: #22d3ee; transform: translateX(3px); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background: linear-gradient(135deg,#0891b2,#0369a1); border-color: #67e8f9;
}}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
[data-testid="stExpander"] {{ background: rgba(8,36,60,.5); border-radius: 14px; border: 1px solid rgba(125,211,252,.18); }}

/* ---- severity badge ---- */
.sev {{ display:inline-block; padding:.25rem .9rem; border-radius:999px; font-weight:700; font-size:.95rem; color:#fff; }}
.sev-Low {{ background:#16a34a; }} .sev-Medium {{ background:#d99a00; }} .sev-High {{ background:#dc2626; }}
.footer {{ text-align:center; color:#7dd3fc; font-size:.78rem; margin-top:2rem; opacity:.75; }}
</style>
""", unsafe_allow_html=True)


def hero(title_plain, title_accent, subtitle):
    st.markdown(f"""
<div class="hero">
  <h1>{title_plain} <span>{title_accent}</span></h1>
  <p>{subtitle}</p>
  <div class="chips">
    <span class="chip">🎨 Enhancement</span><span class="chip">🪨 Crack detection</span>
    <span class="chip">🟠 Corrosion analysis</span><span class="chip">🗺️ Damage map</span>
    <span class="chip">📄 PDF report</span>
  </div>
</div>""", unsafe_allow_html=True)


inject_css()


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


def build_pdf(name, r, cfg=None, **kwargs):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    _, height = A4
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(40, height - 50, "AquaInspect Inspection Report")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, height - 80, f"Image: {name}")

    y = height - 120

    for section in [
        "severity",
        "crack_metrics",
        "corrosion_metrics",
        "timings",
        "damage_pct",
        "refine_info",
    ]:
        values = r.get(section, {})
        if not isinstance(values, dict):
            continue

        if y < 70:
            pdf.showPage()
            y = height - 50

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, y, section.replace("_", " ").title())
        y -= 22
        pdf.setFont("Helvetica", 10)

        for key, value in values.items():
            if isinstance(value, (str, int, float, bool)):
                if y < 45:
                    pdf.showPage()
                    y = height - 50
                    pdf.setFont("Helvetica", 10)

                pdf.drawString(50, y, f"{key}: {value}"[:95])
                y -= 16

        y -= 10

    pdf.save()
    return buffer.getvalue()



def build_csv(name, r):
    rows = []
    def flatten(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(f"{prefix}_{key}" if prefix else key, item)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            rows.append({
                "Image": name,
                "Metric": prefix,
                "Value": value,
            })

    for key in [
        "severity", "crack_metrics", "corrosion_metrics",
        "timings", "damage_pct", "refine_info",
    ]:
        if key in r:
            flatten(key, r[key])

    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")



def summary_row(name, r):
    s = r["severity"]
    cm = r["crack_metrics"]
    cr = r["corrosion_metrics"]

    return {
        "Image": name,
        "Severity_Level": s["severity_level"],
        "Severity_Score": s["severity_score"],
        "Corrosion_%": cr["corrosion_percent"],
        "Crack_Length_px": cm["crack_length_px"],
        "Mean_Crack_Width_px": cm["mean_width_px"],
        "Crack_Index": s["crack_index"],
    }




@st.cache_data(show_spinner=False)
def cached_run(data, name, cfg_json, enhance, refine, min_len, junc, mm_per_px, max_side):
    cfg = json.loads(cfg_json)
    return run_inspection(data, enhance_config=cfg, enhance=enhance, refine=refine,
                        refine_params={"min_len_frac": min_len, "max_junctions": junc},
                        pixel_to_mm=mm_per_px or None, max_side=max_side)


def default_cfg():
    p = os.path.join(RESULTS, "selected_config.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return dict(E.DEFAULT_CONFIG)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🌊 AquaInspect")
st.sidebar.caption("Classical image-processing inspection of underwater pipelines & bridge piers")
page = st.sidebar.radio("Navigate", ["🔍 Inspect", "📊 Batch dashboard", "🧪 Evaluation", "ℹ️ About"], label_visibility="collapsed")

with st.sidebar.expander("⚙️ Pipeline settings", expanded=False):
    dc = default_cfg()
    use_enh = st.checkbox("Apply enhancement", True)
    cfg = {}
    for stg, opts in (("color", E.COLOR_METHODS), ("dehaze", E.DEHAZE_METHODS), ("denoise", E.DENOISE_METHODS), ("contrast", E.CONTRAST_METHODS)):
        cfg[stg] = st.selectbox(f"{stg.capitalize()}", opts, index=opts.index(dc.get(stg, "none")))
    refine = st.checkbox("Crack refinement filter", True, help="Rejects dense texture / short blobs (marine growth, gravel).")
    min_len = st.slider("Min crack length (% of image diagonal)", 1.0, 15.0, 5.0, 0.5) / 100
    junc = st.slider("Max branching (junctions / 100 px of crack)", 2.0, 30.0, 6.0, 0.5, help="Texture & marine growth form branching meshes; cracks rarely do.")
    mm_per_px = st.number_input("Calibration: mm per pixel (0 = unknown)", 0.0, 100.0, 0.0, 0.01)
    max_side = st.select_slider("Max analysed side (px)", [640, 960, 1280, 1600], 1280)
cfg_json = json.dumps(cfg)
args = (cfg_json, use_enh, refine, min_len, junc, mm_per_px, max_side)


# ------------------------------------------------------------------ pages
def page_inspect():
    hero("Underwater", "Defect Inspection", "Upload an image of a submerged pipeline or bridge pier — get cracks, corrosion, a severity score and a damage map.")
    c1, c2 = st.columns([2, 1])
    up = c1.file_uploader("Upload an underwater image (JPG / PNG)", type=["jpg", "jpeg", "png", "bmp"])
    sample = c2.selectbox("…or use a sample image", ["—"] + [os.path.basename(s) for s in SAMPLES])
    data, name = None, None
    if up is not None:
        data, name = up.getvalue(), up.name
    elif sample != "—":
        name = sample
        data = open(os.path.join(ROOT, "data", "sample_images", sample), "rb").read()
    if data is None:
        st.info("Upload an image or pick a sample to start. Typical result time: 1–3 s.")
        return
    try:
        load_image(data)  # early validation -> friendly error
        with st.spinner("Enhancing and analysing…"):
            r = cached_run(data, name, *args)
    except InvalidImageError as e:
        st.error(f"❌ {e}")
        return
    except Exception as e:  # keep the app alive on unexpected failures
        st.error(f"Processing failed: {e}")
        return

    s, cm, cr = r["severity"], r["crack_metrics"], r["corrosion_metrics"]
    lvl = s["severity_level"]
    a, b, c, d, e_ = st.columns(5)
    a.metric("Severity", f"{LEVEL_EMOJI[lvl]} {lvl}", f"score {s['severity_score']:.1f}/100", delta_color="off")
    b.metric("Corrosion coverage", f"{cr['corrosion_percent']:.2f} %")
    c.metric("Crack length", f"{cm['crack_length_px']:.0f} px" + (f" ({cm['crack_length_mm']:.0f} mm)" if cm.get("crack_length_mm") else ""))
    d.metric("Mean crack width", f"{cm['mean_width_px']:.1f} px")
    e_.metric("Processing time", f"{sum(r['timings'].values()):.1f} s")

    st.markdown(f'<div style="margin:.8rem 0 .6rem">Overall condition: <span class="sev sev-{lvl}">{lvl.upper()} — {s["severity_score"]:.1f}/100</span></div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5, t6 = st.tabs(["Original vs Enhanced", "Cracks", "Corrosion", "Damage map", "Enhancement comparison", "Report"])
    with t1:
        x, y = st.columns(2)
        x.image(rgb(r["original"]), caption="Original", width="stretch")
        y.image(rgb(r["enhanced"]), caption="Enhanced", width="stretch")
        sc0, sc1 = M.enhancement_scores(r["original"], r["original"]), M.enhancement_scores(r["original"], r["enhanced"])
        st.dataframe(pd.DataFrame({"Original": [sc0["UIQM"], sc0["UCIQE"], sc0["NoiseSigma"]], "Enhanced": [sc1["UIQM"], sc1["UCIQE"], sc1["NoiseSigma"]]},
                                index=["UIQM ↑", "UCIQE ↑", "Noise σ ↓"]).round(3), width="content")
    with t2:
        x, y, z = st.columns(3)
        x.image(r["crack_mask"], caption="Crack mask (refined)", width="stretch")
        y.image(r["crack_skeleton"], caption="Skeleton (1-px centreline)", width="stretch")
        z.image(rgb(r["crack_overlay"]), caption="Overlay (red mask, yellow skeleton)", width="stretch")
        st.caption(f"Refinement kept {r['refine_info']['kept']} segment(s) and removed {r['refine_info']['removed']} texture/clutter component(s).")
        st.dataframe(pd.DataFrame([{k: v for k, v in cm.items() if v is not None}]), width="stretch", hide_index=True)
    with t3:
        x, y = st.columns(2)
        x.image(r["corrosion_mask"], caption="Corrosion mask (HSV)", width="stretch")
        y.image(rgb(r["corrosion_overlay"]), caption="Corrosion overlay", width="stretch")
        st.write(f"Corrosion area: **{cr['corrosion_pixels']:,} px** ({cr['corrosion_percent']:.2f}% of the image). Green growth and deep shadow are suppressed before measuring.")
    with t4:
        x, y = st.columns([3, 2])
        x.image(rgb(r["damage_map"]), caption="Damage map: 🟢 Low  🟡 Medium  🔴 High", width="stretch")
        y.subheader("Severity breakdown")
        y.dataframe(pd.DataFrame({"Component": ["Corrosion %", "Corrosion index", "Crack density", "Crack index", "Score (0.6·C + 0.4·K)", "Level"],
                                "Value": [s["corrosion_percent"], s["corrosion_index"], s["crack_density"], s["crack_index"], s["severity_score"], lvl]}).astype(str),
                    hide_index=True, width="stretch")
        dp = r["damage_pct"]
        y.bar_chart(pd.DataFrame({"% of map cells": [dp["Low"], dp["Medium"], dp["High"]]}, index=["Low", "Medium", "High"]))
    with t5:
        stg = st.selectbox("Stage to compare", ["color", "dehaze", "denoise", "contrast"])
        opts = [m for m in {"color": E.COLOR_METHODS, "dehaze": E.DEHAZE_METHODS, "denoise": E.DENOISE_METHODS, "contrast": E.CONTRAST_METHODS}[stg] if m != "none"]
        small = E.resize_max(r["original"], 640)
        cols = st.columns(len(opts) + 1)
        cols[0].image(rgb(small), caption="original", width="stretch")
        rows = [{"Method": "original", "UIQM": M.uiqm(small), "UCIQE": M.uciqe(small), "Noise σ": M.noise_sigma(small)}]
        for col, m in zip(cols[1:], opts):
            o = E.apply_stage(small, stg, m)
            col.image(rgb(o), caption=m, width="stretch")
            rows.append({"Method": m, "UIQM": M.uiqm(o), "UCIQE": M.uciqe(o), "Noise σ": M.noise_sigma(o)})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    with t6:
        st.write("Download the inspection report and data for this image.")
        base = os.path.splitext(name)[0]
        x, y, z = st.columns(3)
        x.download_button("📄 PDF report", build_pdf(name, r, {**cfg} if use_enh else {"enhancement": "off"}), f"{base}_report.pdf", "application/pdf")
        y.download_button("📊 Measurements (CSV)", build_csv(name, r), f"{base}_measurements.csv", "text/csv")
        z.download_button("🗺️ Damage map (PNG)", png_bytes(r["damage_map"]), f"{base}_damage_map.png", "image/png")


def page_batch():
    hero("Batch", "Damage Dashboard", "Analyse many images at once and rank them by severity.")
    ups = st.file_uploader("Upload several images (or run the bundled samples)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    go_s = st.button("▶ Run bundled sample set", disabled=bool(ups))
    go_u = st.button("▶ Analyse uploaded images", disabled=not ups)
    if go_s or go_u:
        items = [(u.name, u.getvalue()) for u in ups] if go_u else [(os.path.basename(f), open(f, "rb").read()) for f in SAMPLES]
        rows, results, bar = [], {}, st.progress(0.0)
        for i, (n, d) in enumerate(items):
            try:
                r = cached_run(d, n, *args)
                rows.append(summary_row(n, r)); results[n] = r
            except InvalidImageError as e:
                st.warning(f"Skipped {n}: {e}")
            bar.progress((i + 1) / len(items))
        st.session_state["batch"] = (pd.DataFrame(rows), results)
    if "batch" not in st.session_state:
        return
    df, results = st.session_state["batch"]
    if df.empty:
        st.warning("No valid images.")
        return
    cnt = df["Severity_Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)
    a, b, c, d = st.columns(4)
    a.metric("Images analysed", len(df)); b.metric("🟢 Low", cnt["Low"]); c.metric("🟡 Medium", cnt["Medium"]); d.metric("🔴 High", cnt["High"])
    x, y = st.columns(2)
    x.subheader("Severity distribution"); x.bar_chart(cnt)
    y.subheader("Corrosion % vs crack index"); y.scatter_chart(df, x="Corrosion_%", y="Crack_Index", color="Severity_Level")
    st.subheader("Results table")
    st.dataframe(df.sort_values("Severity_Score", ascending=False), hide_index=True, width="stretch")
    st.download_button("⬇ Download batch results (CSV)", df.to_csv(index=False).encode(), "aquainspect_batch_results.csv", "text/csv")
    st.subheader("Damage maps (most severe first)")
    order = df.sort_values("Severity_Score", ascending=False)["Image"].tolist()
    cols = st.columns(4)
    for i, n in enumerate(order):
        lv = results[n]["severity"]["severity_level"]
        cols[i % 4].image(rgb(results[n]["damage_map"]), caption=f"{LEVEL_EMOJI[lv]} {n} - {lv}", width="stretch")


def show_csv(path, title, note=None):
    if os.path.exists(path):
        st.subheader(title)
        if note:
            st.caption(note)
        st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")


def show_img(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")


def page_eval():
    hero("Evaluation", "Results", "Enhancement comparison, ablation study and detection accuracy.")
    t1, t2, t3, t4 = st.tabs(["Enhancement (real images)", "Synthetic benchmark & ablation", "Detection vs annotated ground truth", "Limitations"])
    with t1:
        st.write("No-reference (UIQM, UCIQE), noise estimate, and PSNR/SSIM *against the input* (measures how strongly a method alters the image, not quality). "
                "Computed over the bundled sample set. Re-create with `python scripts/run_enhancement_eval.py`.")
        show_img(os.path.join(RESULTS, "enhancement_comparison.png"))
        show_csv(os.path.join(RESULTS, "enhancement_summary.csv"), "Mean scores per method")
        show_img(os.path.join(RESULTS, "enhancement_samples.png"), "Each method applied to one sample")
    with t2:
        st.warning("These numbers come from a **synthetic** benchmark (procedural surfaces with exact masks and simulated underwater degradation). "
                   "They validate the pipeline logic and the ablation, but are **not** a measure of real-world accuracy.")
        show_csv(os.path.join(RESULTS, "benchmark_enhancement_fullref.csv"), "Enhancement vs clean reference (true PSNR/SSIM)")
        show_csv(os.path.join(RESULTS, "benchmark_stage_selection.csv"), "Stage-wise selection by downstream detection score")
        show_img(os.path.join(RESULTS, "ablation_study.png"))
        show_csv(os.path.join(RESULTS, "ablation_study.csv"), "Ablation study")
        show_img(os.path.join(RESULTS, "detector_comparison.png"))
        show_csv(os.path.join(RESULTS, "detector_comparison.csv"), "Crack detector comparison")
        show_img(os.path.join(RESULTS, "benchmark_examples.jpg"), "clean | degraded | enhanced | cracks | damage map")
    with t3:
        gt_dir, img_dir = os.path.join(ROOT, "data", "ground_truth"), os.path.join(ROOT, "data", "sample_images")
        files = annotated_images(img_dir, gt_dir)
        st.write("Place manual annotations in `data/ground_truth/` named `<image_stem>_crack.png` / `<image_stem>_corrosion.png` "
                "(white = defect), next to images in `data/sample_images/`.")
        st.info(f"Annotated images found: **{len(files)}**")
        if files and st.button("Run evaluation on annotated images"):
            bar = st.progress(0.0)
            df = evaluate_folder(img_dir, gt_dir, cfg if use_enh else None, enhance=use_enh, refine=refine, progress=bar.progress)
            st.dataframe(df, hide_index=True, width="stretch")
            st.write("**Mean**"); st.dataframe(df.drop(columns="Image").mean().round(4).to_frame("mean").T, width="stretch")
    with t4:
        st.markdown(LIMITATIONS)


LIMITATIONS = """
**Known limitations of the classical approach**
* **Marine growth / algae** – dark or textured growth looks like cracks to edge/black-hat filters; green growth is suppressed for corrosion, but brown/orange biofouling and rust-coloured sediment are still mistaken for corrosion.
* **Shadows and uneven lighting** – deep shadows are removed from the corrosion mask by a brightness cut-off, but shadow boundaries and light-beam edges can still create crack-like lines.
* **Scene clutter** – divers, ROVs, cables, and water-surface ripples produce edges. The refinement filter (length / thinness / isolation) removes most of them, but not all.
* **Thin cracks vs. blur / haze** – very fine cracks (< 2 px) can vanish after denoising or at low resolution.
* **Fixed colour thresholds** – the corrosion hue range was designed for rust-like orange/brown; paint, dyes, or unusual water colour can shift it.
* **Pixel units** – lengths/widths are in pixels unless a camera calibration (mm/px) is supplied; perspective is not corrected.
* **Severity score** – weights (0.6 / 0.4), saturation points and cut-offs (20 / 50) are engineering choices, not calibrated against an inspection standard.
"""


def page_about():
    hero("About", "AquaInspect", "Classical image-processing pipeline for underwater infrastructure inspection.")
    st.markdown("""
**Pipeline**

`Input → validate → Enhancement (colour correction · dehazing · denoising · contrast)` → `Crack detection (grayscale · bilateral filter · black-hat · adaptive threshold · morphology · skeleton · distance transform)`
→ `Corrosion detection (HSV segmentation · morphology · marine-growth & shadow suppression)` → `Severity score` → `Damage map & report`

**Team modules**
* Member 1 – `aquainspect/enhancement.py` (image enhancement)
* Member 2 – `aquainspect/crack_detector.py` (crack detection & measurement; unmodified)
* Member 3 – `aquainspect/corrosion.py` (corrosion detection; unmodified) and the severity formula in `aquainspect/severity.py`
* Member 4 – `app.py`, `aquainspect/pipeline.py`, `crack_refine.py`, `report.py`, `metrics.py`, evaluation scripts

**Severity**: `score = 0.6 × corrosion index + 0.4 × crack index` (each 0-100) → Low < 20 ≤ Medium < 50 ≤ High.
""")
    st.markdown(LIMITATIONS)


{"🔍 Inspect": page_inspect, "📊 Batch dashboard": page_batch, "🧪 Evaluation": page_eval, "ℹ️ About": page_about}[page]()

st.markdown('<div class="footer">AquaInspect · Underwater infrastructure inspection · classical image processing</div>', unsafe_allow_html=True)
