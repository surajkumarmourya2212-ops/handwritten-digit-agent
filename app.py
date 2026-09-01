import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import json
import re


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
# PROFESSIONAL UI STYLE
# ============================================================

st.markdown("""
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
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 0 30px 0;
    border-bottom: 1px solid rgba(148,163,184,0.12);
    margin-bottom: 30px;
}

.logo {
    font-size: 38px;
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
    margin-top: 5px;
}

.ai-badge {
    border: 1px solid rgba(168,85,247,0.35);
    background: rgba(88,28,135,0.15);
    padding: 12px 20px;
    border-radius: 12px;
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
    font-size: 20px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 8px;
}

.card-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 20px;
}


/* ================= UPLOAD ================= */

.upload-box {
    border: 1px dashed rgba(167,139,250,0.55);
    border-radius: 16px;
    padding: 35px 20px;
    text-align: center;
    background: rgba(30,27,75,0.20);
}

.upload-icon {
    font-size: 42px;
}

.upload-text {
    font-size: 18px;
    font-weight: 600;
    margin-top: 10px;
}

.upload-help {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 8px;
}


/* ================= RESULT CARDS ================= */

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


/* ================= EXTRACTED TEXT ================= */

.text-box {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 22px;
    color: #e2e8f0;
    font-size: 18px;
    line-height: 1.8;
    white-space: pre-wrap;
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
""", unsafe_allow_html=True)


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

# Current stable multimodal Gemini model
MODEL = "gemini-3.7-flash"


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header">

    <div>
        <div class="logo">
            <span class="logo-icon">✍️</span>
            OCR Handwritten
        </div>

        <div class="tagline">
            AI-Powered Handwritten Text & Digit Recognition
        </div>
    </div>

    <div class="ai-badge">
        ✨ Powered by Gemini AI
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns([0.85, 1.5], gap="large")


# ============================================================
# LEFT SIDE
# ============================================================

with left:

    st.markdown("""
    <div class="card">

        <div class="card-title">
            📤 Upload Handwritten Image
        </div>

        <div class="card-subtitle">
            Upload an image containing handwritten
            text, digits, or both.
        </div>

    </div>
    """, unsafe_allow_html=True)


    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed"
    )


    st.markdown("""
    <div class="card">

        <div class="card-title">
            💡 Tips
        </div>

        <div style="color:#94a3b8; line-height:2;">

        • Use a clear handwritten image<br>
        • Good lighting gives better results<br>
        • Supports handwritten text<br>
        • Supports handwritten digits<br>
        • Supports text + digits together<br>
        • Random/non-handwritten images are rejected

        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RIGHT SIDE
# ============================================================

with right:

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.markdown("""
        <div class="card">

            <div class="card-title">
                🖼️ Uploaded Image
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True
        )

        analyze_button = st.button(
            "🔍  Analyze Image",
            type="primary",
            use_container_width=True
        )

    else:

        st.markdown("""
        <div class="card" style="text-align:center; padding:80px 30px;">

            <div style="font-size:55px;">
                ✍️
            </div>

            <div style="
                font-size:24px;
                font-weight:700;
                margin-top:15px;
            ">
                Ready to recognize handwriting
            </div>

            <div style="
                color:#94a3b8;
                margin-top:10px;
            ">
                Upload an image to begin OCR analysis.
            </div>

        </div>
        """, unsafe_allow_html=True)

        analyze_button = False


# ============================================================
# AI ANALYSIS
# ============================================================

if uploaded_file and analyze_button:

    with st.spinner(
        "🧠 Gemini AI is analyzing your handwriting..."
    ):

        try:

            # Convert image to PNG bytes
            image_bytes = io.BytesIO()

            image.save(
                image_bytes,
                format="PNG"
            )

            image_data = image_bytes.getvalue()


            # ==================================================
            # OCR + VALIDATION PROMPT
            # ==================================================

            prompt = """
You are an advanced handwritten OCR system.

Analyze this image carefully.

Your task is to recognize ONLY meaningful handwritten
content visible in the image.

The image can contain:

1. Handwritten digits
2. Handwritten text
3. Handwritten digits and text together
4. No valid handwritten content

IMPORTANT RULES:

- Do NOT guess missing characters.
- Do NOT hallucinate text.
- Do NOT convert random objects into text.
- Do NOT treat printed UI text as handwritten content.
- If the image is a random object, landscape, animal,
  room, laptop, screenshot, blank image, etc.,
  mark it INVALID.
- If there is handwritten text and digits together,
  extract BOTH.
- Preserve the reading order.
- Return the handwriting as accurately as possible.

Return ONLY JSON.

Use exactly this structure:

{
  "valid": true,
  "content_type": "digits_and_text",
  "extracted_text": "My Roll No is 12345",
  "digits_found": ["1","2","3","4","5"],
  "confidence": "high",
  "message": "Handwritten text and digits detected successfully."
}

content_type MUST be one of:

"digits"
"text"
"digits_and_text"
"invalid"

For invalid images return:

{
  "valid": false,
  "content_type": "invalid",
  "extracted_text": "",
  "digits_found": [],
  "confidence": "high",
  "message": "No valid handwritten text or digits detected."
}
"""


            # ==================================================
            # SEND IMAGE TO GEMINI
            # ==================================================

            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=image_data,
                        mime_type="image/png"
                    ),
                    prompt
                ]
            )


            raw = response.text.strip()


            # Remove markdown JSON formatting
            raw = re.sub(
                r"```json",
                "",
                raw,
                flags=re.IGNORECASE
            )

            raw = raw.replace("```", "").strip()


            result = json.loads(raw)


            # ==================================================
            # RESULT
            # ==================================================

            st.markdown("---")

            st.markdown(
                "## 🎯 Recognition Result"
            )


            # ==================================================
            # INVALID IMAGE
            # ==================================================

            if not result.get("valid", False):

                st.markdown("""
                <div class="invalid-box">

                    <h3>❌ Invalid Image</h3>

                    No valid handwritten text or digits
                    were detected in this image.

                </div>
                """, unsafe_allow_html=True)


                st.info(
                    result.get(
                        "message",
                        "Please upload a handwritten image."
                    )
                )


            # ==================================================
            # VALID IMAGE
            # ==================================================

            else:

                content_type = result.get(
                    "content_type",
                    "unknown"
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
                    "unknown"
                )

                message = result.get(
                    "message",
                    "Content recognized successfully."
                )


                # ==================================================
                # SUMMARY CARDS
                # ==================================================

                r1, r2, r3 = st.columns(3)


                with r1:

                    st.markdown(
                        f"""
                        <div class="result-card">

                            <div class="result-label">
                                Content Type
                            </div>

                            <div class="result-value">
                                {content_type.replace("_", " ").title()}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                with r2:

                    st.markdown(
                        f"""
                        <div class="result-card">

                            <div class="result-label">
                                AI Confidence
                            </div>

                            <div class="result-value">
                                {confidence.title()}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                with r3:

                    st.markdown(
                        f"""
                        <div class="result-card">

                            <div class="result-label">
                                Digits Detected
                            </div>

                            <div class="result-value">
                                {len(digits)}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # ==================================================
                # EXTRACTED CONTENT
                # ==================================================

                st.markdown("### 📝 Extracted Content")

                st.markdown(
                    f"""
                    <div class="text-box">
{extracted_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ==================================================
                # DIGITS
                # ==================================================

                if digits:

                    st.markdown("### 🔢 Detected Digits")

                    digit_text = " • ".join(
                        str(d) for d in digits
                    )

                    st.success(
                        f"Digits detected: {digit_text}"
                    )


                # ==================================================
                # AI EXPLANATION
                # ==================================================

                st.markdown("### 🧠 AI Explanation")

                st.info(message)


                # ==================================================
                # SUCCESS
                # ==================================================

                st.markdown("""
                <div class="success-box">

                    <strong>
                        ✅ Handwritten content recognized successfully!
                    </strong>

                    <br><br>

                    The AI identified and extracted
                    the handwritten content from your image.

                </div>
                """, unsafe_allow_html=True)


                # ==================================================
                # DOWNLOAD RESULT
                # ==================================================

                download_text = f"""
OCR HANDWRITTEN
============================

Content Type:
{content_type}

AI Confidence:
{confidence}

Extracted Content:
{extracted_text}

Detected Digits:
{", ".join(str(x) for x in digits)}

AI Explanation:
{message}

============================
Powered by Gemini AI
"""


                st.download_button(
                    "⬇️ Download Result",
                    data=download_text,
                    file_name="ocr_handwritten_result.txt",
                    mime="text/plain"
                )


        except json.JSONDecodeError:

            st.error(
                "⚠️ Gemini returned an unexpected format."
            )

            with st.expander(
                "Show technical response"
            ):
                st.code(raw)


        except Exception as e:

            st.error(
                "❌ AI processing error"
            )

            with st.expander(
                "Show technical details"
            ):
                st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    ✍️ <strong>OCR Handwritten</strong>
    &nbsp; • &nbsp;
    AI-Powered Handwritten Text & Digit Recognition
    <br><br>

    Built with Streamlit + Gemini AI

</div>
""", unsafe_allow_html=True)
