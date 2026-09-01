import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- MAIN APP ---------- */

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
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- HEADER ---------- */

    .hero {
        text-align: center;
        padding: 25px 10px 35px 10px;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        color: white;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 16px;
    }

    .ai-badge {
        display: inline-block;
        margin-top: 18px;
        padding: 10px 20px;
        border-radius: 30px;

        border: 1px solid rgba(168,85,247,0.45);

        background: rgba(88,28,135,0.20);

        color: #ddd6fe;

        font-weight: 700;
    }


    /* ---------- GENERAL CARD ---------- */

    .card {
        background: rgba(15,23,42,0.78);

        border: 1px solid rgba(148,163,184,0.16);

        border-radius: 20px;

        padding: 25px;

        margin-bottom: 20px;

        box-shadow:
            0 15px 45px rgba(0,0,0,0.18);
    }

    .card-title {
        font-size: 22px;
        font-weight: 750;
        color: white;
        margin-bottom: 10px;
    }

    .card-text {
        color: #cbd5e1;
        font-size: 15px;
        line-height: 1.7;
    }


    /* ---------- FEATURE ---------- */

    .feature {
        background: rgba(30,41,59,0.55);

        border: 1px solid rgba(148,163,184,0.12);

        border-radius: 15px;

        padding: 16px;

        text-align: center;
    }

    .feature-name {
        color: #94a3b8;
        font-size: 13px;
    }

    .feature-value {
        color: white;
        font-size: 19px;
        font-weight: 700;
        margin-top: 6px;
    }


    /* ---------- EMPTY STATE ---------- */

    .empty-card {
        background: rgba(15,23,42,0.78);

        border: 1px solid rgba(148,163,184,0.16);

        border-radius: 20px;

        min-height: 360px;

        padding: 50px 25px;

        text-align: center;
    }

    .empty-icon {
        font-size: 60px;
        margin-top: 40px;
    }

    .empty-title {
        color: white;
        font-size: 25px;
        font-weight: 750;
        margin-top: 15px;
    }

    .empty-text {
        color: #94a3b8;
        margin-top: 10px;
    }


    /* ---------- RESULT ---------- */

    .result-box {
        background: #020617;

        border: 1px solid #1e293b;

        border-radius: 15px;

        padding: 22px;

        color: #e2e8f0;

        font-size: 18px;

        line-height: 1.8;

        white-space: pre-wrap;

        word-break: break-word;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;

        color: #64748b;

        margin-top: 50px;

        padding-top: 25px;

        border-top:
            1px solid rgba(148,163,184,0.10);

        font-size: 13px;
    }

    </style>
    """,
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
        "Go to Streamlit Secrets and add GEMINI_API_KEY."
    )

    st.stop()


client = genai.Client(
    api_key=API_KEY
)


# Current Gemini model
MODEL = "gemini-3.7-flash"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            ✍️ OCR Handwritten
        </div>

        <div class="hero-subtitle">
            AI-Powered Handwritten Text &amp; Digit Recognition
        </div>

        <div class="ai-badge">
            ✨ Powered by Gemini AI
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INTRODUCTION CARD
# ============================================================

st.markdown(
    """
    <div class="card">

        <div class="card-title">
            📄 Intelligent Handwriting Recognition
        </div>

        <div class="card-text">
            Upload an image containing handwritten text,
            handwritten digits, or both.

            Gemini AI analyzes the image,
            validates the handwriting,
            and extracts the content.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FEATURES
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        <div class="feature">

            <div class="feature-name">
                ✍️ Handwriting
            </div>

            <div class="feature-value">
                Supported
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="feature">

            <div class="feature-name">
                🔢 Digits
            </div>

            <div class="feature-value">
                Supported
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="feature">

            <div class="feature-name">
                📝 Text + Digits
            </div>

            <div class="feature-value">
                Supported
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns(
    [0.9, 1.35],
    gap="large"
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left:

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                📤 Upload Image
            </div>

            <div class="card-text">
                PNG, JPG, JPEG or WEBP.
                Clear handwriting gives better results.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        label_visibility="collapsed"
    )


    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                💡 Best Results
            </div>

            <div class="card-text">

                ✅ Use a clear handwritten image
                <br>

                ✅ Keep good lighting
                <br>

                ✅ Avoid extremely blurry images
                <br>

                ✅ Text and digits can appear together
                <br>

                ✅ Printed/random images can be rejected

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with right:

    if uploaded_file is None:

        st.markdown(
            """
            <div class="empty-card">

                <div class="empty-icon">
                    ✍️
                </div>

                <div class="empty-title">
                    Ready for OCR
                </div>

                <div class="empty-text">
                    Upload a handwritten image
                    to start analysis.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        analyze_button = False

    else:

        try:

            image = Image.open(
                uploaded_file
            )

            image.load()


            st.markdown(
                """
                <div class="card-title">
                    🖼️ Image Preview
                </div>
                """,
                unsafe_allow_html=True
            )


            st.image(
                image,
                use_container_width=True
            )


            analyze_button = st.button(
                "🔍 Analyze Handwriting",
                type="primary",
                use_container_width=True
            )


        except Exception:

            st.error(
                "❌ Could not read this image."
            )

            analyze_button = False


# ============================================================
# AI ANALYSIS
# ============================================================

if uploaded_file is not None and analyze_button:

    prompt = """
You are a strict handwritten OCR system.

Analyze the supplied image.

Accept ONLY meaningful handwritten text
or handwritten digits.

Do NOT guess missing characters.

Do NOT hallucinate content.

Reject the image if it is:

- a random object
- a landscape
- an animal
- a room
- a laptop
- a computer screenshot
- a blank image
- printed-only content
- an image with no meaningful handwriting

If handwriting contains both text and digits,
extract both.

Preserve the reading order.

Return ONLY valid JSON.

Use exactly this structure:

{
    "valid": true,
    "content_type": "digits_and_text",
    "extracted_text": "My Roll No is 12345",
    "digits_found": ["1","2","3","4","5"],
    "confidence": "high",
    "message": "Handwritten text and digits detected successfully."
}

content_type must be exactly one of:

"digits"

"text"

"digits_and_text"

"invalid"

For an invalid image return:

{
    "valid": false,
    "content_type": "invalid",
    "extracted_text": "",
    "digits_found": [],
    "confidence": "high",
    "message": "No valid handwritten text or digits detected."
}
"""


    with st.spinner(
        "🧠 Gemini AI is analyzing your handwriting..."
    ):

        try:

            response = client.models.generate_content(
                model=MODEL,

                contents=[
                    image,
                    prompt
                ],

                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
            )


            raw = response.text.strip()


            #
