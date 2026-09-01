import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import json
import re
import textwrap
import html


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="OCR Handwritten",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# HELPER FOR HTML
# IMPORTANT: prevents indented HTML from becoming code
# ============================================================

def ui_html(content):
    return textwrap.dedent(content).strip()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    ui_html("""
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(76, 29, 149, 0.22),
                transparent 32%
            ),
            radial-gradient(
                circle at 90% 0%,
                rgba(30, 64, 175, 0.20),
                transparent 30%
            ),
            #070b16;
        color: #f8fafc;
    }

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ================= HEADER ================= */

    .header {
        text-align: center;
        padding: 20px 0 35px 0;
        margin-bottom: 25px;
    }

    .logo {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1.5px;
        color: white;
    }

    .logo-icon {
        color: #a855f7;
    }

    .tagline {
        color: #94a3b8;
        font-size: 16px;
        margin-top: 8px;
    }

    .ai-badge {
        display: inline-block;
        margin-top: 20px;
        border: 1px solid rgba(168,85,247,0.40);
        background: rgba(88,28,135,0.20);
        padding: 10px 20px;
        border-radius: 30px;
        color: #ddd6fe;
        font-weight: 600;
    }

    /* ================= CARDS ================= */

    .card {
        background: rgba(15,23,42,0.75);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 15px 45px rgba(0,0,0,0.18);
    }

    .card-title {
        font-size: 21px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .card-subtitle {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 18px;
    }

    /* ================= INFO CARD ================= */

    .info-card {
        background: rgba(15,23,42,0.70);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 25px;
    }

    .info-title {
        font-size: 24px;
        font-weight: 750;
        color: white;
        margin-bottom: 10px;
    }

    .info-description {
        color: #cbd5e1;
        line-height: 1.7;
        font-size: 15px;
    }

    /* ================= FEATURE BOXES ================= */

    .feature {
        background: rgba(30,41,59,0.55);
        border-radius: 14px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(148,163,184,0.10);
    }

    .feature-icon {
        font-size: 22px;
    }

    .feature-name {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 5px;
    }

    .feature-value {
        color: white;
        font-size: 18px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* ================= EMPTY STATE ================= */

    .empty-state {
        background: rgba(15,23,42,0.75);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 18px;
        padding: 80px 25px;
        text-align: center;
        min-height: 300px;
    }

    .empty-icon {
        font-size: 55px;
    }

    .empty-title {
        color: white;
        font-size: 25px;
        font-weight: 700;
        margin-top: 15px;
    }

    .empty-description {
        color: #94a3b8;
        margin-top: 10px;
        font-size: 15px;
    }

    /* ================= RESULT ================= */

    .result-card {
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 16px;
        padding: 20px;
        min-height: 120px;
    }

    .result-label {
        color: #94a3b8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }

    .result-value {
        color: white;
        font-size: 22px;
        font-weight: 750;
        margin-top: 12px;
    }

    /* ================= TEXT ================= */

    .text-box {
        background: #020617;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 22px;
        color: #e2e8f0;
        font-size: 18px;
        line-height: 1.8;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* ================= TIPS ================= */

    .tip {
        color: #cbd5e1;
        line-height: 2;
        font-size: 15px;
    }

    /* ================= SUCCESS ================= */

    .success-box {
        border: 1px solid rgba(34,197,94,0.30);
        background: rgba(22,101,52,0.15);
        color: #86efac;
        padding: 18px;
        border-radius: 14px;
        margin-top: 20px;
    }

    /* ================= INVALID ================= */

    .invalid-box {
        border: 1px solid rgba(239,68,68,0.30);
        background: rgba(127,29,29,0.15);
        color: #fca5a5;
        padding: 20px;
        border-radius: 14px;
    }

    /* ================= FOOTER ================= */

    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 50px;
        padding-top: 25px;
        border-top: 1px solid rgba(148,163,184,0.10);
        font-size: 13px;
    }

    </style>
    """),
    unsafe_allow_html=True
)


# ============================================================
# GEMINI API
# ============================================================

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = None


if not API_KEY:
    st.error("🔐 Gemini API key is missing.")
    st.info(
        "Open Streamlit Settings → Secrets and add "
        "GEMINI_API_KEY."
    )
    st.stop()


client = genai.Client(api_key=API_KEY)

# Use the model mentioned by the error you received
MODEL = "gemini-3.6-flash"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    ui_html("""
    <div class="header">

        <div class="logo">
            <span class="logo-icon">✍️</span>
            OCR Handwritten
        </div>

        <div class="tagline">
            AI-Powered Handwritten Text & Digit Recognition
        </div>

        <div class="ai-badge">
            ✨ Powered by Gemini AI
        </div>

    </div>
    """),
    unsafe_allow_html=True
)
