# -*- coding: utf-8 -*-
# app.py - Streamlit Web Application
# Run: streamlit run app.py
# cd "C:\Users\User\Desktop\Refurb AI\smartphone_classifier"
# streamlit run app.py

import io
import os
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Refurb AI",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0d0d0d; color: #e8e8e8; }
    #MainMenu, footer, header { visibility: hidden; }
    .hero { text-align: center; padding: 3rem 0 1.5rem; }
    .hero-badge {
        display: inline-block; font-family: 'Space Mono', monospace;
        font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase;
        color: #00e5a0; border: 1px solid #00e5a0; border-radius: 2px;
        padding: 3px 10px; margin-bottom: 1rem;
    }
    .hero h1 { font-family: 'Space Mono', monospace; font-size: 2.6rem; font-weight: 700; color: #fff; margin: 0; }
    .hero h1 span { color: #00e5a0; }
    .hero p { color: #888; font-size: 0.95rem; margin-top: 0.75rem; }
    .card { background: #111; border: 1px solid #1e1e1e; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
    .card-title { font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase; color: #444; margin-bottom: 0.75rem; }
    .verdict-good { font-family: 'Space Mono', monospace; font-size: 1.7rem; font-weight: 700; color: #00e5a0; }
    .verdict-bad  { font-family: 'Space Mono', monospace; font-size: 1.7rem; font-weight: 700; color: #ff4c6e; }
    .verdict-sub  { font-size: 0.82rem; color: #555; margin-top: 0.2rem; }
    .metric-row   { display: flex; gap: 1rem; margin-top: 1rem; }
    .metric-box   { flex: 1; background: #0d0d0d; border: 1px solid #1e1e1e; border-radius: 6px; padding: 0.9rem 1rem; }
    .metric-label { font-family: 'Space Mono', monospace; font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase; color: #444; margin-bottom: 0.3rem; }
    .metric-value { font-family: 'Space Mono', monospace; font-size: 1.1rem; color: #e8e8e8; }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #00e5a0, #00b07a) !important; border-radius: 2px !important; }
    .stProgress > div > div { background: #1a1a1a !important; border-radius: 2px !important; height: 6px !important; }
    .stButton > button { width: 100%; background: #00e5a0; color: #0d0d0d; border: none; border-radius: 4px; font-family: 'Space Mono', monospace; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; padding: 0.75rem 1.5rem; cursor: pointer; }
    .stButton > button:hover { background: #00c98a; }
    .footer { text-align: center; font-family: 'Space Mono', monospace; font-size: 0.6rem; letter-spacing: 0.12em; color: #2a2a2a; padding: 3rem 0 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def get_model():
    from predict import load_trained_model
    return load_trained_model()

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">AI &middot; Computer Vision &middot; MobileNetV2</div>
        <h1>Phone<span>Guard</span></h1>
        <p>Upload a smartphone image &mdash; detect physical damage in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = "model/model.h5"
model_exists = os.path.isfile(MODEL_PATH)

if not model_exists:
    st.warning("No trained model found. Run `python train.py` first.")

st.markdown('<p style="font-family:monospace;font-size:0.7rem;color:#555;letter-spacing:0.15em;text-transform:uppercase;">Upload Image</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(label="", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    st.markdown('<div class="card"><p class="card-title">Uploaded Image</p>', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("ANALYZE IMAGE"):
        if not model_exists:
            st.error("Model not found. Please run `python train.py` first.")
        else:
            with st.spinner("Running inference ..."):
                model = get_model()
                from predict import predict
                result = predict(img, model=model)

            label      = result["label"]
            confidence = result["confidence"]
            scores     = result["all_scores"]
            is_good    = label == "not_damaged"

            st.markdown('<div class="card"><p class="card-title">Analysis Result</p>', unsafe_allow_html=True)

            if is_good:
                st.markdown('<div class="verdict-good">Good Condition</div><p class="verdict-sub">The device appears to be undamaged.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-bad">Damaged</div><p class="verdict-sub">Physical damage detected on the device.</p>', unsafe_allow_html=True)

            st.markdown(
                '<div class="metric-row">'
                '<div class="metric-box"><div class="metric-label">Prediction</div><div class="metric-value">{}</div></div>'
                '<div class="metric-box"><div class="metric-label">Confidence</div><div class="metric-value">{}</div></div>'
                '</div>'.format(label.replace("_", " ").upper(), "{:.1%}".format(confidence)),
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card"><p class="card-title">Score Breakdown</p>', unsafe_allow_html=True)
            for cls_name, score in scores.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown('<span style="font-size:0.78rem;color:#666;">{}</span>'.format(cls_name.replace("_", " ").title()), unsafe_allow_html=True)
                    st.progress(float(score))
                with col2:
                    st.markdown('<div style="text-align:right;font-family:monospace;font-size:0.85rem;padding-top:0.25rem;">{:.1%}</div>'.format(score), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="footer">PHONEGUARD AI &middot; MOBILENETV2 &middot; TENSORFLOW</div>', unsafe_allow_html=True)
