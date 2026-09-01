import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import json


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
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(99, 102, 241, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 0%,
                rgba(139, 92, 246, 0.14),
                transparent 30%
            ),
            #080b14;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ---------- TITLE ---------- */

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }


    /* ---------- BADGE ---------- */

    .badge-row {
        text-align: center;
        margin-bottom: 35px;
    }

    .badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 30px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(129, 140, 248, 0.30);
        color: #c7d2fe;
        font-size: 0.9rem;
        font-weight: 600;
    }


    /* ---------- SECTION TITLE ---------- */

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .section-description {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }


    /* ---------- INFO CARDS ---------- */

    .info-card {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.13);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
    }


    /* ---------- EMPTY STATE ---------- */

    .empty-state {
        text-align: center;
        padding: 70px 20px;
        border: 1px dashed rgba(148, 163, 184, 0.25);
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.45);
    }

    .empty-icon {
        font-size: 55px;
        margin-bottom: 12px;
    }

    .empty-title {
        font-size: 1.5rem;
        font-weight: 700;
    }

    .empty-description {
        color: #94a3b8;
        margin-top: 8px;
    }


    /* ---------- RESULT ---------- */

    .result-heading {
        font-size: 1.7rem;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 15px;
    }


    /* ---------- EXTRACTED TEXT ---------- */

    .extracted {
        background: #0b1020;
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 25px;
        font-size: 1.15rem;
        line-height: 1.8;
        min-height: 120px;
        white-space: pre-wrap;
    }


    /* ---------- SUCCESS ---------- */

    .success {
        padding: 18px;
        border-radius: 15px;
        background: rgba(34, 197, 94, 0.10);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #86efac;
        margin-top: 18px;
    }


    /* ---------- INVALID ---------- */

    .invalid {
        padding: 22px;
        border-radius: 15px;
        background: rgba(239, 68, 68, 0.10);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #fca5a5;
        margin-top: 18px;
    }


    /* ---------- FOOTER ---------- */

    .project-footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding-top: 35px;
        margin-top: 40px;
        border-top: 1px solid rgba(148,163,184,0.10);
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
        "Go to your Streamlit app → Settings → Secrets "
        "and add GEMINI_API_KEY."
    )

    st.stop()


client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.7-flash"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">✍️ OCR Handwritten</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Handwritten Text & Digit Recognition'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="badge-row">'
    '<span class="badge">✨ Powered by Gemini AI</span>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PROJECT INTRODUCTION
# ============================================================

with st.container(border=True):

    st.markdown("### 📋 Intelligent Handwriting Recognition")

    st.write(
        "Upload an image containing handwritten text, digits, "
        "or a combination of both. Gemini AI analyzes the image, "
        "validates the handwriting, and extracts the content."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("✍️ Handwriting", "Supported")

    with c2:
        st.metric("🔢 Digits", "Supported")

    with c3:
        st.metric("🔤 Text + Digits", "Supported")


# ============================================================
# MAIN COLUMNS
# ============================================================

left, right = st.columns([0.9, 1.3], gap="large")


# ============================================================
# LEFT COLUMN
# ============================================================

with left:

    st.markdown("### 📤 Upload Image")

    st.caption(
        "PNG, JPG, JPEG or WEBP • Clear handwriting gives better results"
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed"
    )


    st.markdown("### 💡 Best Results")

    with st.container(border=True):

        st.write("✅ Use a clear handwritten image")

        st.write("✅ Keep good lighting")

        st.write("✅ Avoid extremely blurry images")

        st.write("✅ Text and digits can appear together")

        st.write("✅ Printed/random images can be rejected")


# ============================================================
# RIGHT COLUMN
# ============================================================

with right:

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.markdown("### 🖼️ Image Preview")

        st.image(
            image,
            use_container_width=True
        )

        analyze = st.button(
            "🔍 Analyze Handwriting",
            type="primary",
            use_container_width=True
        )

    else:

        st.markdown(
            """
            <div class="empty-state">

                <div class="empty-icon">✍️</div>

                <div class="empty-title">
                    Ready for OCR
                </div>

                <div class="empty-description">
                    Upload a handwritten image to start analysis.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        analyze = False


# ============================================================
# AI ANALYSIS
# ============================================================

if uploaded_file and analyze:

    st.markdown("---")

    with st.spinner("🧠 Gemini AI is analyzing the handwriting..."):

        try:

            # ------------------------------------------------
            # Convert image to bytes
            # ------------------------------------------------

            image_buffer = io.BytesIO()

            image.save(
                image_buffer,
                format="PNG"
            )

            image_bytes = image_buffer.getvalue()


            # ------------------------------------------------
            # OCR INSTRUCTIONS
            # ------------------------------------------------

            prompt = """
You are an advanced handwriting OCR and image validation system.

Analyze the uploaded image carefully.

YOUR PRIMARY TASK:

Determine whether the image contains genuine handwritten
letters, handwritten digits, or both.

The system must NOT blindly extract text from every image.

VALID INPUTS:

1. Handwritten digits
2. Handwritten words or sentences
3. Handwritten digits + text together

INVALID INPUTS:

1. Random photographs
2. People
3. Animals
4. Buildings
5. Food
6. Nature
7. Screenshots without handwritten content
8. Printed-only documents
9. Blank images
10. Images where handwriting cannot reasonably be identified

IMPORTANT:

- Only recognize content that appears handwritten.
- Do not treat printed text as handwriting.
- Do not invent missing characters.
- Do not guess unclear words.
- Preserve the original reading order.
- If text and numbers appear together, recognize both.
- A number inside handwritten text must remain a number.
- If there is no meaningful handwriting, return INVALID.
- Do not describe objects in the image as OCR text.

For confidence, use:
"High", "Medium", or "Low".

For content_type use exactly one of:

"digits"
"text"
"digits_and_text"
"invalid"

For valid handwriting:

- extracted_text = complete recognized handwriting
- digits_found = every digit appearing in the handwriting
- message = short explanation of what was detected

For invalid images:

- extracted_text = ""
- digits_found = []
- confidence = "High"
- message = explain that no valid handwritten content was detected
"""


            # ------------------------------------------------
            # STRUCTURED JSON SCHEMA
            # ------------------------------------------------

            response_schema = {
                "type": "OBJECT",
                "properties": {

                    "valid": {
                        "type": "BOOLEAN"
                    },

                    "content_type": {
                        "type": "STRING",
                        "enum": [
                            "digits",
                            "text",
                            "digits_and_text",
                            "invalid"
                        ]
                    },

                    "extracted_text": {
                        "type": "STRING"
                    },

                    "digits_found": {
                        "type": "ARRAY",
                        "items": {
                            "type": "STRING"
                        }
                    },

                    "confidence": {
                        "type": "STRING",
                        "enum": [
                            "High",
                            "Medium",
                            "Low"
                        ]
                    },

                    "message": {
                        "type": "STRING"
                    }
                },

                "required": [
                    "valid",
                    "content_type",
                    "extracted_text",
                    "digits_found",
                    "confidence",
                    "message"
                ]
            }


            # ------------------------------------------------
            # SEND TO GEMINI
            # ------------------------------------------------

            response = client.models.generate_content(

                model=MODEL,

                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png"
                    ),

                    prompt
                ],

                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema
                }
            )


            # ------------------------------------------------
            # GET RESULT
            # ------------------------------------------------

            result = json.loads(response.text)


            valid = result.get(
                "valid",
                False
            )

            content_type = result.get(
                "content_type",
                "invalid"
            )

            extracted_text = result.get(
                "extracted_text",
                ""
            )

            digits = result.get(
                "digits_found",
                []
            )

            confidence = result.get(
                "confidence",
                "Low"
            )

            message = result.get(
                "message",
                ""
            )


            # =================================================
            # RESULT
            # =================================================

            st.markdown(
                '<div class="result-heading">🎯 Recognition Result</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # INVALID
            # =================================================

            if (
                not valid
                or content_type == "invalid"
                or not extracted_text.strip()
            ):

                st.markdown(
                    """
                    <div class="invalid">

                        <h3>❌ Invalid Image</h3>

                        No valid handwritten text or digits
                        were detected.

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.info(
                    message
                    if message
                    else
                    "Please upload an image containing handwriting."
                )


            # =================================================
            # VALID
            # =================================================

            else:

                # ------------------------------------------------
                # SUMMARY
                # ------------------------------------------------

                r1, r2, r3 = st.columns(3)

                with r1:

                    st.metric(
                        "Content Type",
                        content_type.replace(
                            "_",
                            " "
                        ).title()
                    )

                with r2:

                    st.metric(
                        "AI Confidence",
                        confidence
                    )

                with r3:

                    st.metric(
                        "Digits Found",
                        len(digits)
                    )


                # ------------------------------------------------
                # EXTRACTED TEXT
                # ------------------------------------------------

                st.markdown("### 📝 Extracted Handwriting")

                st.text_area(
                    "Recognized content",
                    value=extracted_text,
                    height=160,
                    label_visibility="collapsed"
                )


                # ------------------------------------------------
                # DIGITS
                # ------------------------------------------------

                if digits:

                    st.markdown("### 🔢 Detected Digits")

                    digit_string = "  •  ".join(
                        str(d)
                        for d in digits
                    )

                    st.success(
                        f"Digits detected: {digit_string}"
                    )


                # ------------------------------------------------
                # AI DECISION
                # ------------------------------------------------

                st.markdown("### 🧠 AI Analysis")

                st.info(message)


                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                st.markdown(
                    """
                    <div class="success">

                        <strong>
                        ✅ Handwritten content recognized successfully.
                        </strong>

                        <br>

                        The AI detected valid handwritten content
                        and extracted it from the uploaded image.

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ------------------------------------------------
                # DOWNLOAD
                # ------------------------------------------------

                download_content = f"""
OCR HANDWRITTEN
==============================

Content Type:
{content_type}

AI Confidence:
{confidence}

Extracted Handwriting:
{extracted_text}

Detected Digits:
{", ".join(digits)}

AI Analysis:
{message}

==============================
Powered by Gemini AI
"""


                st.download_button(
                    label="⬇️ Download OCR Result",
                    data=download_content,
                    file_name="ocr_handwritten_result.txt",
                    mime="text/plain",
                    use_container_width=True
                )


        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except Exception as e:

            st.error(
                "⚠️ Unable to analyze the image."
            )

            with st.expander(
                "Technical details"
            ):

                st.code(
                    str(e)
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="project-footer">

        ✍️ <strong>OCR Handwritten</strong>

        <br>

        AI-Powered Handwritten Text & Digit Recognition

        <br><br>

        Built with Streamlit + Gemini AI

    </div>
    """,
    unsafe_allow_html=True
)
