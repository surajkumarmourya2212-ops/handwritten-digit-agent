import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image
from google import genai
from google.genai import types


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic AI Handwritten Digit Recognition",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Agentic AI Handwritten Digit Recognition")

st.write(
    "Upload an image containing ONE handwritten digit (0–9)."
)

st.info(
    "The AI Agent first checks whether the image is a valid "
    "handwritten digit. Only then is it sent to the CNN model."
)


# ============================================================
# LOAD YOUR EXISTING CNN MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "digit_model.keras"
    )


model = load_model()


# ============================================================
# CONNECT TO GEMINI
# ============================================================

@st.cache_resource
def load_gemini():

    api_key = st.secrets["GEMINI_API_KEY"]

    client = genai.Client(
        api_key=api_key
    )

    return client


gemini = load_gemini()


# ============================================================
# GEMINI IMAGE VALIDATION
# ============================================================

def validate_image_with_gemini(image):

    prompt = """
You are the validation agent for a handwritten digit
recognition system.

Carefully analyze the image.

The image is VALID only when it contains exactly ONE clearly
visible handwritten digit from 0 to 9.

VALID:
- One handwritten digit on paper
- A clear photo of one handwritten 0–9
- One handwritten digit with a simple background

INVALID:
- Cats, dogs, people, food, vehicles, buildings, objects
- Screenshots of websites or applications
- Computer screens
- Code
- Paragraphs
- Words
- Sentences
- Multiple digits
- Multiple handwritten characters
- Blank images
- Random objects
- Printed text
- A digit that is part of unrelated content

Do NOT guess.

Return ONLY JSON in this exact format:

{
    "is_valid": true,
    "reason": "One handwritten digit is clearly visible",
    "candidate_digit": 7
}

For an invalid image return:

{
    "is_valid": false,
    "reason": "The image does not contain one handwritten digit",
    "candidate_digit": null
}

candidate_digit must be an integer from 0 to 9 when valid.
"""

    try:

        response = gemini.models.generate_content(
            model="gemini-3.7-flash",
            contents=[
                prompt,
                image
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        result = json.loads(response.text)

        return result

    except Exception as e:

        return {
            "is_valid": False,
            "reason": f"Gemini validation failed: {str(e)}",
            "candidate_digit": None
        }


# ============================================================
# IMAGE PREPROCESSING FOR YOUR CNN
# ============================================================

def preprocess_digit_image(image):

    image_array = np.array(image)

    # RGB → grayscale
    if len(image_array.shape) == 3:

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

    else:

        gray = image_array


    # Remove noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # Convert black handwriting on white paper
    # into white digit on black background
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )


    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # Remove tiny noise
    contours = [
        c for c in contours
        if cv2.contourArea(c) > 20
    ]


    if not contours:

        return None


    # Largest contour
    contour = max(
        contours,
        key=cv2.contourArea
    )


    # Bounding box
    x, y, w, h = cv2.boundingRect(
        contour
    )


    # Add padding
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


    # ========================================================
    # MAKE IMAGE SQUARE
    # ========================================================

    h, w = cropped.shape

    size = max(
        h,
        w
    )


    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )


    y_offset = (
        size - h
    ) // 2

    x_offset = (
        size - w
    ) // 2


    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = cropped


    # ========================================================
    # RESIZE TO 28 × 28
    # ========================================================

    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )


    # ========================================================
    # NORMALIZE
    # ========================================================

    normalized = (
        resized.astype("float32") / 255.0
    )


    return normalized.reshape(
        1,
        28,
        28
    )


# ============================================================
# CNN PREDICTION
# ============================================================

def predict_digit(image):

    processed = preprocess_digit_image(
        image
    )


    if processed is None:

        return None, 0.0


    probabilities = model.predict(
        processed,
        verbose=0
    )[0]


    digit = int(
        np.argmax(
            probabilities
        )
    )


    confidence = float(
        np.max(
            probabilities
        )
    )


    return digit, confidence


# ============================================================
# AGENT
# ============================================================

def digit_agent(image):

    # --------------------------------------------------------
    # STEP 1 — GEMINI VALIDATES IMAGE
    # --------------------------------------------------------

    validation = validate_image_with_gemini(
        image
    )


    # --------------------------------------------------------
    # INVALID IMAGE
    # --------------------------------------------------------

    if not validation.get(
        "is_valid",
        False
    ):

        return {
            "status": "invalid",
            "reason": validation.get(
                "reason",
                "This is not a valid handwritten digit image."
            )
        }


    # --------------------------------------------------------
    # STEP 2 — CNN PREDICTION
    # --------------------------------------------------------

    digit, confidence = predict_digit(
        image
    )


    if digit is None:

        return {
            "status": "invalid",
            "reason": "The digit could not be extracted from the image."
        }


    # Gemini's candidate digit
    gemini_digit = validation.get(
        "candidate_digit"
    )


    # --------------------------------------------------------
    # STEP 3 — AGENT COMPARES GEMINI AND CNN
    # --------------------------------------------------------

    try:

        gemini_digit = int(
            gemini_digit
        )

    except:

        gemini_digit = None


    # --------------------------------------------------------
    # HIGH CONFIDENCE + AGREEMENT
    # --------------------------------------------------------

    if (
        confidence >= 0.90
        and gemini_digit == digit
    ):

        return {
            "status": "accepted",
            "digit": digit,
            "confidence": confidence,
            "gemini_digit": gemini_digit,
            "decision": (
                "Gemini and CNN agree. "
                "High-confidence prediction accepted."
            )
        }


    # --------------------------------------------------------
    # HIGH CNN CONFIDENCE BUT DISAGREEMENT
    # --------------------------------------------------------

    if confidence >= 0.90:

        return {
            "status": "review",
            "digit": digit,
            "confidence": confidence,
            "gemini_digit": gemini_digit,
            "decision": (
                "CNN has high confidence, but the AI validator "
                "and CNN disagree. Manual review is recommended."
            )
        }


    # --------------------------------------------------------
    # MEDIUM CONFIDENCE
    # --------------------------------------------------------

    if confidence >= 0.70:

        return {
            "status": "review",
            "digit": digit,
            "confidence": confidence,
            "gemini_digit": gemini_digit,
            "decision": (
                "A digit was detected, but CNN confidence is "
                "moderate. Please review the prediction."
            )
        }


    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------

    return {
        "status": "uncertain",
        "digit": digit,
        "confidence": confidence,
        "gemini_digit": gemini_digit,
        "decision": (
            "Prediction confidence is too low. "
            "Please upload a clearer handwritten digit."
        )
    }


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ============================================================
# RUN AGENT
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
        width=400
    )


    if st.button(
        "🤖 Analyze Image"
    ):

        with st.spinner(
            "AI Agent is analyzing..."
        ):

            result = digit_agent(
                image
            )


        # ====================================================
        # INVALID
        # ====================================================

        if result["status"] == "invalid":

            st.error(
                "❌ INVALID IMAGE"
            )

            st.write(
                result["reason"]
            )

            st.info(
                "Please upload a clear image containing "
                "ONE handwritten digit from 0 to 9."
            )


        # ====================================================
        # ACCEPTED
        # ====================================================

        elif result["status"] == "accepted":

            st.success(
                f"✅ Predicted Digit: {result['digit']}"
            )

            st.metric(
                "CNN Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.write(
                f"👁️ Gemini Digit: "
                f"{result['gemini_digit']}"
            )

            st.success(
                f"🤖 Agent Decision: "
                f"{result['decision']}"
            )


        # ====================================================
        # REVIEW
        # ====================================================

        elif result["status"] == "review":

            st.warning(
                f"⚠️ Predicted Digit: {result['digit']}"
            )

            st.metric(
                "CNN Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.write(
                f"👁️ Gemini Digit: "
                f"{result['gemini_digit']}"
            )

            st.warning(
                f"🤖 Agent Decision: "
                f"{result['decision']}"
            )


        # ====================================================
        # UNCERTAIN
        # ====================================================

        elif result["status"] == "uncertain":

            st.error(
                "❌ Prediction Not Reliable"
            )

            st.write(
                f"CNN predicted: **{result['digit']}**"
            )

            st.metric(
                "CNN Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.info(
                f"🤖 Agent Decision: "
                f"{result['decision']}"
            )
