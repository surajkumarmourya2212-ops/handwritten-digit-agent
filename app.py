import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import json
import re
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
# GLOBAL CSS
# ============================================================

st.html("""
<style>

    /* ========================================================
       MAIN APP
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(79, 70, 229, 0.22),
                transparent 32%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(14, 165, 233, 0.15),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #020617 0%,
                #0f172a 50%,
                #111827 100%
            );

        color: #f8fafc;
    }


    /* ========================================================
       HIDE STREAMLIT DEFAULT ELEMENTS
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ========================================================
       PAGE WIDTH
       ======================================================== */

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .ocr-header {
        text-align: center;
        padding: 20px 10px 35px 10px;
    }

    .ocr-logo {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1.5px;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .ocr-logo-icon {
        font-size: 42px;
        margin-right: 8px;
    }

    .ocr-tagline {
        font-size: 16px;
        color: #94a3b8;
        margin-top: 5px;
    }

    .gemini-badge {
        display: inline-block;
        margin-top: 20px;
        padding: 10px 22px;
        border-radius: 30px;

        background:
            linear-gradient(
                135deg,
                rgba(79,70,229,0.22),
                rgba(14,165,233,0.18)
            );

        border: 1px solid rgba(129,140,248,0.45);

        color: #e0e7ff;
        font-size: 14px;
        font-weight: 700;

        box-shadow:
            0 8px 30px rgba(30,64,175,0.18);
    }


    /* ========================================================
       INFORMATION CARD
       ======================================================== */

    .info-card {
        background:
            linear-gradient(
                135deg,
                rgba(30,41,59,0.85),
                rgba(15,23,42,0.78)
            );

        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 22px;

        padding: 30px;
        margin-bottom: 25px;

        box-shadow:
            0 20px 60px rgba(0,0,0,0.20);
    }

    .info-title {
        font-size: 25px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .info-description {
        font-size: 15px;
        line-height: 1.7;
        color: #cbd5e1;
        margin-bottom: 25px;
    }


    /* ========================================================
       FEATURE BOXES
       ======================================================== */

    .feature-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
    }

    .feature-box {
        background: rgba(15,23,42,0.75);
        border: 1px solid rgba(148,163,184,0.12);

        border-radius: 16px;
        padding: 18px;

        text-align: center;
   
