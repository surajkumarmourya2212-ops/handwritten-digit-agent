import streamlit as st
import numpy as np
import cv2
from PIL import Image
from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Digit & Text AI",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Handwritten Digit & Text AI")
st.write(
    "Upload an image containing handwritten digits, text, or both."
)


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():

    if "GEMINI_API_KEY" not in st.secrets:
        return None

    api_key = st.secrets["GEMINI_API_KEY"]

    return genai.Client(api_key=api_key)


gemini_client = get_gemini_client()


# ============================================================
# OPTIONAL DIGIT MODEL
# ============================================================

@st.cache_resource
def load_digit_model():

    try:

        import tensorflow as tf

        model = tf.keras.models.load_model(
            "digit_model.keras"
        )

        return model

    except Exception:

        return None


digit_model = load_digit_model()


# ============================================================
# IMAGE QUALITY CHECK
# ============================================================

def basic_image_quality_check(image):

    image_array = np.array(image)

    if image_array.size == 0:
        return False, "The image is empty."

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    # Very dark or very bright / blank image
    mean_value = np.mean(gray)
    standard_deviation = np.std(gray)

    if standard_deviation < 5:

        return False, "The image appears blank or has very little information."

    if mean_value < 3:

        return False, "The image is almost completely black."

    if mean_value > 252:

        return False, "The image is almost completely white."

    return True, "Image quality looks acceptable."


# ============================================================
# PREPROCESS FOR OLD DIGIT MODEL
# ============================================================

def preprocess_digit_image(image):

    image_array = np.array(image)

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    # Remove small noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Convert black handwriting on white paper
    # into white handwriting on black background
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        # Ignore extremely tiny objects
        contours = [
            c for c in contours
            if cv2.contourArea(c) > 20
        ]

    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        padding = int(
            max(w, h) * 0.25
        )

        x1 = max(
            0,
            x - padding
        )

        y1 = max(
            0,
            y - padding
        )

        x2 = min(
            binary.shape[1],
            x + w + padding
        )

        y2 = min(
            binary.shape[0],
            y + h + padding
        )

        cropped = binary[
            y1:y2,
            x1:x2
        ]

    else:

        cropped = binary


    # --------------------------------------------------------
    # MAKE IMAGE SQUARE
    # --------------------------------------------------------

    h, w = cropped.shape

    size = max(
        h,
        w
    )

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = cropped


    # --------------------------------------------------------
    # RESIZE TO MNIST SIZE
    # --------------------------------------------------------

    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    normalized = (
        resized.astype("float32") / 255.0
    )

    return normalized.reshape(
        1,
        28,
        28
    )


# ============================================================
# DIGIT MODEL PREDICTION
# ============================================================

def predict_with_digit_model(image):

    if digit_model is None:

        return None, None

    try:

        processed = preprocess_digit_image(
            image
        )

        probabilities = digit_model.predict(
            processed,
            verbose=0
        )[0]

        digit = int(
            np.argmax(probabilities)
        )

        confidence = float(
            np.max(probabilities)
        )

        return digit, confidence

    except Exception:

        return None, None


# ============================================================
# GEMINI IMAGE ANALYSIS
# ============================================================

def analyze_with_gemini(image):

    if gemini_client is None:

        return {
            "success": False,
            "message": (
                "Gemini API key was not found. "
                "Please add GEMINI_API_KEY to Streamlit Secrets."
            )
        }


    try:

        # Convert PIL image to JPEG bytes
        image_bytes = __import__(
            "io"
        ).BytesIO()

        image.save(
            image_bytes,
            format="JPEG"
        )

        image_data = image_bytes.getvalue()


        # Create Gemini image part
        image_part = types.Part.from_bytes(
            data=image_data,
            mime_type="image/jpeg"
        )


        prompt = """
You are the image validation and handwriting recognition agent
for a Handwritten Digit and Text Recognition application.

Analyze the uploaded image carefully.

Your job is to determine:

1. Is this image valid for handwriting recognition?
2. Does it contain handwritten content?
3. Does it contain:
   - digits
   - handwritten text
   - both digits and text
   - no handwriting
4. If handwriting exists, transcribe ONLY the visible handwritten
   content as accurately as possible.
5. Do NOT invent text that is not visible.
6. If the image is a random photo, screenshot, object, landscape,
   computer screen, blank paper, printed document, or unrelated image,
   mark it as INVALID unless clear handwritten content is visible.
7. If the handwriting is unclear, say that it is unclear rather than
   guessing.
8. If there are multiple lines, preserve the approximate line order.

Return your answer exactly in this format:

VALID: YES or NO

TYPE: DIGIT / TEXT / BOTH / NONE

TRANSCRIPTION:
Write the detected handwritten content here.
If there is no readable handwriting, write NONE.

REASON:
Give one short reason.

CONFIDENCE:
Give a number from 0 to 100.

IMPORTANT:
Do not force a digit prediction.
Do not assume every image contains a digit.
Reject unrelated images.
"""


        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash-lite",
            contents=[
                prompt,
                image_part
            ]
        )


        text = response.text

        return {
            "success": True,
            "response": text
        }


    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


# ============================================================
# PARSE GEMINI RESPONSE
# ============================================================

def parse_gemini_response(text):

    result = {
        "valid": "UNKNOWN",
        "type": "UNKNOWN",
        "transcription": "UNKNOWN",
        "reason": "",
        "confidence": None
    }


    lines = text.splitlines()

    current_section = None

    transcription_lines = []


    for line in lines:

        clean = line.strip()

        upper = clean.upper()


        if upper.startswith("VALID:"):

            result["valid"] = (
                clean.split(
                    ":",
                    1
                )[1]
                .strip()
                .upper()
            )


        elif upper.startswith("TYPE:"):

            result["type"] = (
                clean.split(
                    ":",
                    1
                )[1]
                .strip()
                .upper()
            )


        elif upper.startswith("TRANSCRIPTION:"):

            current_section = "transcription"

            value = clean.split(
                ":",
                1
            )[1].strip()

            if value:
                transcription_lines.append(
                    value
                )


        elif upper.startswith("REASON:"):

            current_section = "reason"

            result["reason"] = clean.split(
                ":",
                1
            )[1].strip()


        elif upper.startswith("CONFIDENCE:"):

            current_section = "confidence"

            value = clean.split(
                ":",
                1
            )[1].strip()

            try:

                value = (
                    value
                    .replace("%", "")
                    .strip()
                )

                result["confidence"] = float(
                    value
                )

            except:

                result["confidence"] = None


        elif current_section == "transcription":

            if clean:

                transcription_lines.append(
                    clean
                )


        elif current_section == "reason":

            if clean:

                result["reason"] += " " + clean


    if transcription_lines:

        result["transcription"] = "\n".join(
            transcription_lines
        )


    return result


# ============================================================
# AGENT
# ============================================================

def handwriting_agent(image):

    # --------------------------------------------------------
    # STEP 1 — BASIC IMAGE CHECK
    # --------------------------------------------------------

    valid, quality_message = (
        basic_image_quality_check(image)
    )


    if not valid:

        return {
            "status": "invalid",
            "message": quality_message
        }


    # --------------------------------------------------------
    # STEP 2 — GEMINI VALIDATION
    # --------------------------------------------------------

    gemini_result = analyze_with_gemini(
        image
    )


    if not gemini_result["success"]:

        return {
            "status": "error",
            "message": gemini_result["message"]
        }


    # --------------------------------------------------------
    # STEP 3 — PARSE GEMINI
    # --------------------------------------------------------

    parsed = parse_gemini_response(
        gemini_result["response"]
    )


    # --------------------------------------------------------
    # STEP 4 — REJECT INVALID IMAGE
    # --------------------------------------------------------

    if parsed["valid"] == "NO":

        return {
            "status": "invalid",
            "type": parsed["type"],
            "reason": parsed["reason"],
            "confidence": parsed["confidence"]
        }


    # --------------------------------------------------------
    # STEP 5 — DIGIT MODEL CROSS-CHECK
    # --------------------------------------------------------

    digit_prediction = None
    digit_confidence = None


    if parsed["type"] == "DIGIT":

        digit_prediction, digit_confidence = (
            predict_with_digit_model(image)
        )


    # --------------------------------------------------------
    # STEP 6 — FINAL RESULT
    # --------------------------------------------------------

    return {
        "status": "success",
        "type": parsed["type"],
        "transcription": parsed["transcription"],
        "reason": parsed["reason"],
        "gemini_confidence": parsed["confidence"],
        "digit_prediction": digit_prediction,
        "digit_confidence": digit_confidence
    }


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload your image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.subheader(
        "📷 Uploaded Image"
    )

    st.image(
        image,
        use_container_width=True
    )


    if st.button(
        "🤖 Analyze Image",
        type="primary"
    ):

        with st.spinner(
            "AI Agent is analyzing the image..."
        ):

            result = handwriting_agent(
                image
            )


        # ====================================================
        # INVALID IMAGE
        # ====================================================

        if result["status"] == "invalid":

            st.error(
                "❌ Invalid Image / No suitable handwriting detected."
            )


            if "reason" in result:

                st.warning(
                    f"Reason: {result['reason']}"
                )


            if "confidence" in result:

                if result["confidence"] is not None:

                    st.write(
                        f"AI confidence: "
                        f"{result['confidence']:.2f}%"
                    )


        # ====================================================
        # ERROR
        # ====================================================

        elif result["status"] == "error":

            st.error(
                "⚠️ AI processing error"
            )

            st.write(
                result["message"]
            )


        # ====================================================
        # SUCCESS
        # ====================================================

        else:

            st.success(
                "✅ Handwritten content detected!"
            )


            # ------------------------------------------------
            # TYPE
            # ------------------------------------------------

            st.subheader(
                "🔎 Detected Type"
            )

            st.info(
                result["type"]
            )


            # ------------------------------------------------
            # TRANSCRIPTION
            # ------------------------------------------------

            st.subheader(
                "✍️ Recognized Content"
            )

            st.success(
                result["transcription"]
            )


            # ------------------------------------------------
            # GEMINI CONFIDENCE
            # ------------------------------------------------

            if result["gemini_confidence"] is not None:

                st.metric(
                    "AI Confidence",
                    f"{result['gemini_confidence']:.2f}%"
                )


            # ------------------------------------------------
            # DIGIT MODEL
            # ------------------------------------------------

            if result["digit_prediction"] is not None:

                st.subheader(
                    "🔢 Digit Model Cross-Check"
                )

                st.write(
                    f"Digit Model Prediction: "
                    f"**{result['digit_prediction']}**"
                )

                st.write(
                    f"Digit Model Confidence: "
                    f"**{result['digit_confidence'] * 100:.2f}%**"
                )


            # ------------------------------------------------
            # AGENT DECISION
            # ------------------------------------------------

            st.subheader(
                "🤖 Agent Decision"
            )

            if result["type"] == "DIGIT":

                st.info(
                    "The image contains a handwritten digit. "
                    "Gemini analyzed the image and the local "
                    "digit model was used as an additional check."
                )

            elif result["type"] == "TEXT":

                st.info(
                    "The image contains handwritten text. "
                    "Gemini was used to analyze and transcribe it."
                )

            elif result["type"] == "BOTH":

                st.info(
                    "The image contains both handwritten "
                    "digits and text. Gemini analyzed the mixed content."
                )

            else:

                st.info(
                    "The image was analyzed as handwritten content."
                )


            # ------------------------------------------------
            # REASON
            # ------------------------------------------------

            if result["reason"]:

                st.subheader(
                    "🧠 AI Explanation"
                )

                st.write(
                    result["reason"]
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 Handwritten Digit & Text Recognition "
    "powered by Streamlit + Gemini AI"
)
