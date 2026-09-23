"""
app/streamlit_app.py

PAPULONET: AI-Assisted Papulosquamous Skin Disease Analysis
Production-grade research demonstration interface for the AEF-CRC multimodal pipeline.

Architecture:
  Streamlit UI -> modules.app_adapter.run_app_inference() -> Frozen Phase 9/10 Pipeline

Scientific Contract & Boundaries:
- Operates strictly in-memory; user images are never persisted to project datasets.
- Phase 3-10 models, weights, masks, calibrators, and conformal thresholds are frozen.
- Explains the two-tier division of responsibility (Grad-CAM on CNN, TreeSHAP on RF).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Configure Streamlit page layout
st.set_page_config(
    page_title="PAPULONET | AEF-CRC",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for clean, professional medical-AI research styling
CUSTOM_CSS = """
<style>
    /* Main container and font styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 1.75rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0;
        color: #f8fafc;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.25rem;
        margin-bottom: 0.75rem;
        font-weight: 400;
    }
    .disclaimer-box {
        background-color: rgba(245, 158, 11, 0.12);
        border-left: 4px solid #f59e0b;
        color: #fbbf24;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }

    /* Metric & result cards */
    .result-card {
        background-color: var(--background-secondary, #f8fafc);
        border: 1px solid var(--border-color, #e2e8f0);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .status-badge-standard {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-review {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fcd34d;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-alert {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Conformal set tag */
    .conformal-tag {
        background-color: #e0e7ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.85rem;
        margin-right: 0.4rem;
        display: inline-block;
        margin-top: 0.25rem;
    }

    /* Section titles */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-color, #1e293b);
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Subtext descriptions */
    .desc-text {
        font-size: 0.88rem;
        color: #64748b;
        margin-bottom: 0.75rem;
        line-height: 1.45;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Frozen class names & metadata
FROZEN_CLASSES = [
    "Psoriasis",
    "Lichen_Planus",
    "Pityriasis_Rosea",
    "Seborrheic_Dermatitis",
]
DEFAULT_HANDOFF_PATH = Path("artifacts/phase9_v2/final_pipeline_handoff.joblib")


@st.cache_resource(show_spinner=False)
def get_cached_pipeline(handoff_path_str: str, device: str = "auto"):
    """Loads and caches the production pipeline in memory."""
    from modules.app_adapter import get_app_pipeline
    return get_app_pipeline(handoff_or_path=Path(handoff_path_str), device=device, use_cache=True)


def init_session_state() -> None:
    """Initializes session state variables for multiple image support and navigation."""
    if "images" not in st.session_state:
        st.session_state.images = []  # List of dicts: {id, name, bytes, pil, result, error}
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0


init_session_state()

# -----------------------------------------------------------------------------
# 1. HEADER SECTION
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-card">
        <div class="header-title">PAPULONET</div>
        <div class="header-subtitle">AI-Assisted Papulosquamous Skin Disease Analysis (AEF-CRC Multimodal Framework)</div>
        <div class="disclaimer-box">
            <strong>Research & Demonstration Disclaimer:</strong> This system is designed for scientific research and educational 
            evaluation of multimodal clinical AI. The generated predictions, calibration metrics, and conformal bounds do not constitute 
            a medical diagnosis and must always be evaluated by a certified dermatologist.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. IMAGE INPUT & UPLOAD SECTION
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">1. Clinical Lesion Input</div>', unsafe_allow_html=True)

input_tab1, input_tab2 = st.tabs(["📁 Upload Lesion Images", "📷 Capture via Camera"])

with input_tab1:
    uploaded_files = st.file_uploader(
        "Upload one or more clinical lesion photographs (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="You may select multiple images. Analysis is conducted strictly in-memory without saving patient data.",
    )
    if uploaded_files:
        # Check if new files were added
        existing_names = {img["name"] for img in st.session_state.images}
        new_files = [f for f in uploaded_files if f.name not in existing_names]
        for f in new_files:
            file_bytes = f.read()
            try:
                pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                st.session_state.images.append({
                    "id": f"img_{len(st.session_state.images) + 1:03d}",
                    "name": f.name,
                    "bytes": file_bytes,
                    "pil": pil_img,
                    "result": None,
                    "error": None,
                })
            except Exception as exc:
                st.error(f"Failed to decode image '{f.name}': {exc}")

with input_tab2:
    camera_file = st.camera_input("Take a clinical photo")
    if camera_file is not None:
        cam_bytes = camera_file.read()
        cam_name = f"camera_capture_{len(st.session_state.images) + 1:03d}.jpg"
        if not any(img["name"] == cam_name for img in st.session_state.images):
            try:
                cam_pil = Image.open(io.BytesIO(cam_bytes)).convert("RGB")
                st.session_state.images.append({
                    "id": f"img_{len(st.session_state.images) + 1:03d}",
                    "name": cam_name,
                    "bytes": cam_bytes,
                    "pil": cam_pil,
                    "result": None,
                    "error": None,
                })
            except Exception as exc:
                st.error(f"Failed to decode camera capture: {exc}")

# Reset/Clear button
col_clear, col_info = st.columns([2, 8])
with col_clear:
    if st.session_state.images:
        if st.button("🗑️ Clear All Uploads", use_container_width=True):
            st.session_state.images = []
            st.session_state.current_idx = 0
            st.rerun()

# -----------------------------------------------------------------------------
# 3. NAVIGATION & BATCH QUEUE DISPLAY
# -----------------------------------------------------------------------------
total_images = len(st.session_state.images)

if total_images == 0:
    st.info("Please upload or capture clinical lesion images to begin analysis.")
    st.stop()

# Ensure current index is valid
st.session_state.current_idx = max(0, min(st.session_state.current_idx, total_images - 1))
current_item = st.session_state.images[st.session_state.current_idx]

st.markdown('<div class="section-title">2. Image Inspection & Analysis Queue</div>', unsafe_allow_html=True)

# Navigation toolbar
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([2, 3, 2, 3])

with nav_col1:
    prev_disabled = (st.session_state.current_idx == 0)
    if st.button("⬅️ Previous Image", disabled=prev_disabled, use_container_width=True):
        st.session_state.current_idx -= 1
        st.rerun()

with nav_col2:
    st.markdown(
        f"<div style='text-align: center; padding-top: 8px; font-weight: 600;'>"
        f"Image {st.session_state.current_idx + 1} of {total_images} &mdash; <code>{current_item['name']}</code>"
        f"</div>",
        unsafe_allow_html=True,
    )

with nav_col3:
    next_disabled = (st.session_state.current_idx >= total_images - 1)
    if st.button("Next Image ➡️", disabled=next_disabled, use_container_width=True):
        st.session_state.current_idx += 1
        st.rerun()

with nav_col4:
    analyze_all_btn = st.button("⚡ Analyze All Pending", use_container_width=True)

# Main action row
col_preview, col_action = st.columns([4, 6])

with col_preview:
    st.image(
        current_item["pil"],
        caption=f"{current_item['name']} ({current_item['pil'].width}x{current_item['pil'].height})",
        use_container_width=True,
    )

with col_action:
    st.write(f"**Image Identifier**: `{current_item['id']}`")
    st.write(f"**File**: `{current_item['name']}`")
    st.write(f"**Resolution**: {current_item['pil'].width} &times; {current_item['pil'].height} px")
    
    # Analysis trigger
    analyze_btn = st.button(
        "🔬 Analyze Current Image",
        type="primary",
        use_container_width=True,
    )

# Execute inference if requested
from modules.app_adapter import AppInferenceError, run_app_inference

items_to_analyze = []
if analyze_btn:
    items_to_analyze.append(current_item)
elif analyze_all_btn:
    items_to_analyze = [item for item in st.session_state.images if item["result"] is None]

if items_to_analyze:
    # Ensure pipeline is loaded with spinner
    with st.spinner("Initializing frozen AEF-CRC pipeline and loading checkpoint weights..."):
        try:
            pipeline = get_cached_pipeline(str(DEFAULT_HANDOFF_PATH), device="auto")
        except Exception as exc:
            st.error(f"Failed to load production pipeline handoff: {exc}")
            st.stop()

    for item in items_to_analyze:
        with st.spinner(f"Analyzing {item['name']} (EfficientNet-B0 + A7 Fusion + BDA + Temperature Scaling + Conformal + XAI)..."):
            try:
                res = run_app_inference(
                    image_input=item["bytes"],
                    handoff_path=DEFAULT_HANDOFF_PATH,
                    pipeline=pipeline,
                )
                item["result"] = res
                item["error"] = None
            except AppInferenceError as err:
                item["error"] = str(err)
                item["result"] = None
            except Exception as exc:
                item["error"] = f"Unexpected inference exception: {exc}"
                item["result"] = None
    st.rerun()

# -----------------------------------------------------------------------------
# 4. RESULTS DISPLAY SECTION
# -----------------------------------------------------------------------------
if current_item["error"]:
    st.error(f"⚠️ Analysis Failure: {current_item['error']}")
    st.stop()

res = current_item["result"]
if res is None:
    st.info("Click **'Analyze Current Image'** to run classification, probability calibration, conformal prediction, and explainability.")
    st.stop()

st.markdown('<div class="section-title">3. Diagnostic Assessment & Conformal Bounds</div>', unsafe_allow_html=True)

# Top metrics summary
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.metric(
        label="Predicted Primary Disease",
        value=res.predicted_class.replace("_", " "),
        help="Top-1 point prediction derived from Temperature Scaling-calibrated probabilities.",
    )

with m_col2:
    st.metric(
        label="Calibrated Confidence",
        value=f"{res.calibrated_confidence * 100:.1f}%",
        help="Post-processed probability via Phase 8 Temperature Scaling.",
    )

with m_col3:
    st.metric(
        label="Raw RF Confidence",
        value=f"{res.raw_confidence * 100:.1f}%",
        help="Diagnostic only: Uncalibrated ensemble vote proportion from the Random Forest.",
    )

with m_col4:
    # Review status badge
    status = res.review_status
    if status == "STANDARD_OUTPUT":
        badge_html = '<span class="status-badge-standard">STANDARD OUTPUT</span>'
    elif status == "SPECIALIST_REVIEW_REQUIRED":
        badge_html = '<span class="status-badge-review">SPECIALIST REVIEW REQUIRED</span>'
    else:
        badge_html = f'<span class="status-badge-alert">{status}</span>'
    
    st.markdown(
        f"<div style='font-size: 0.85rem; color: #64748b;'>Clinical Review Triage</div>{badge_html}",
        unsafe_allow_html=True,
    )

# Explanation of review status
st.markdown(
    f"<div class='desc-text'><strong>Triage Assessment:</strong> {res.review_status_explanation}</div>",
    unsafe_allow_html=True,
)

# Conformal prediction breakdown
conf_col1, conf_col2 = st.columns(2)

with conf_col1:
    st.markdown("#### 90% Nominal Marginal Prediction Set")
    if res.marginal_prediction_set:
        tags = "".join([f'<span class="conformal-tag">{cls.replace("_", " ")}</span>' for cls in res.marginal_prediction_set])
        st.markdown(tags, unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge-alert">Empty Set &empty;</span>', unsafe_allow_html=True)
    st.caption(r"Guaranteed finite-sample $\ge 90\%$ marginal coverage under exchangeability. Set size: " + str(len(res.marginal_prediction_set)))

with conf_col2:
    st.markdown("#### Mondrian Class-Conditional Prediction Set")
    if res.mondrian_prediction_set:
        tags = "".join([f'<span class="conformal-tag">{cls.replace("_", " ")}</span>' for cls in res.mondrian_prediction_set])
        st.markdown(tags, unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge-alert">Empty Set &empty;</span>', unsafe_allow_html=True)
    st.caption("Class-conditional coverage controlling for individual disease prevalence.")

# Probability distribution table & chart
st.markdown("#### Calibrated Class Probability Distribution")
prob_data = []
for cls in FROZEN_CLASSES:
    prob_data.append({
        "Disease Class": cls.replace("_", " "),
        "Calibrated Probability (%)": round(res.calibrated_probabilities.get(cls, 0.0) * 100, 2),
        "Raw Ensemble Vote (%)": round(res.raw_probabilities.get(cls, 0.0) * 100, 2),
    })
prob_df = pd.DataFrame(prob_data)

p_col1, p_col2 = st.columns([5, 5])
with p_col1:
    st.dataframe(prob_df, hide_index=True, use_container_width=True)

with p_col2:
    chart_df = pd.DataFrame({
        "Probability (%)": [res.calibrated_probabilities.get(cls, 0.0) * 100 for cls in FROZEN_CLASSES]
    }, index=[cls.replace("_", " ") for cls in FROZEN_CLASSES])
    st.bar_chart(chart_df, horizontal=True)

# -----------------------------------------------------------------------------
# 5. EXPLAINABILITY SECTION (TWO-TIER RIGOR)
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">4. Multimodal Explainable AI (XAI)</div>', unsafe_allow_html=True)

xai_tab_gradcam, xai_tab_shap = st.tabs([
    "🔍 Grad-CAM — CNN Visual Explanation",
    "📊 TreeSHAP — Random Forest Feature Explanation",
])

# -----------------------------------------------------------------------------
# 5A. Grad-CAM Section
# -----------------------------------------------------------------------------
with xai_tab_gradcam:
    st.markdown("### Grad-CAM &mdash; CNN Visual Explanation")
    st.markdown(
        """
        <div class="desc-text">
            <strong>Scientific Boundary:</strong> Grad-CAM computes gradient-weighted class activation maps targeting the 
            <strong>pre-softmax logit</strong> of the Phase 3 EfficientNet-B0 disease head. It identifies spatial morphological 
            lesion patterns contributing to the visual feature extraction layer. 
            <em>It does not directly explain the Random Forest classifier, Temperature Scaling probability calibration, or conformal prediction boundaries.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    g_col1, g_col2, g_col3 = st.columns(3)

    with g_col1:
        st.markdown("**1. Original Lesion (Standardized)**")
        st.image(current_item["pil"], use_container_width=True)

    with g_col2:
        st.markdown("**2. Spatial Heatmap (JET)**")
        if res.gradcam_heatmap_pil:
            st.image(res.gradcam_heatmap_pil, use_container_width=True)
        else:
            st.caption("Heatmap unavailable.")

    with g_col3:
        st.markdown("**3. Grad-CAM Overlay**")
        if res.gradcam_overlay_pil:
            st.image(res.gradcam_overlay_pil, use_container_width=True)
        else:
            st.caption("Overlay unavailable.")

    st.info(f"Targeted Pre-Softmax Logit: `{res.gradcam_logit:.4f}` for predicted class `{res.predicted_class}`")

# -----------------------------------------------------------------------------
# 5B. TreeSHAP Section
# -----------------------------------------------------------------------------
with xai_tab_shap:
    st.markdown("### TreeSHAP &mdash; Random Forest Feature Explanation")
    st.markdown(
        """
        <div class="desc-text">
            <strong>Scientific Boundary:</strong> TreeSHAP applies path-dependent feature perturbation on the production Random Forest 
            across the <strong>642 BDA-selected features</strong>. It decomposes the model's <strong>raw uncalibrated ensemble prediction</strong> 
            into branch-level and feature-level attributions. 
            <em>TreeSHAP explains the Random Forest decisions, not the CNN backbone, Temperature Scaling calibrator, or conformal prediction thresholds.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top branch summary
    st.markdown(
        f"**Dominant Multimodal Branch**: `'{res.top_shap_branch}'` "
        f"contributing **{res.top_shap_branch_pct:.1f}%** of total attribution magnitude."
    )

    s_col1, s_col2 = st.columns([5, 5])

    with s_col1:
        st.markdown("#### Multimodal Branch Importance Breakdown")
        if not res.branch_importance_df.empty:
            st.dataframe(res.branch_importance_df, hide_index=True, use_container_width=True)
        else:
            st.caption("Branch breakdown unavailable.")

    with s_col2:
        st.markdown("#### Branch Attribution Magnitude")
        if not res.branch_importance_df.empty:
            b_chart = pd.DataFrame(
                res.branch_importance_df["pct_total_importance"].values,
                index=res.branch_importance_df["branch"].values,
                columns=["Attribution (%)"],
            )
            st.bar_chart(b_chart, horizontal=True)

    st.markdown("#### Top Individual Feature Attributions (from 642-D Selected Subspace)")
    if not res.top_features_df.empty:
        st.dataframe(res.top_features_df.head(15), hide_index=True, use_container_width=True)
    else:
        st.caption("Individual feature contributions unavailable.")

# -----------------------------------------------------------------------------
# 6. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "AEF-CRC Phase 10 V2 Production Deployment &bull; Model Provenance: `efficientnet_b0_43d581b96f8ec368` "
    "&bull; 1316-D Multimodal Representation (A7) &bull; 642-D BDA Mask &bull; Temperature Scaling &bull; 90% Nominal Split-Conformal Coverage"
)
