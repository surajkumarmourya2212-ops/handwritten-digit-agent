import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.keras")


model = load_model()


# ==========================================
# PREPROCESS IMAGE
# ==========================================

def preprocess_digit_image(image):

    image = np.array(image)

    # RGB → grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = image

    # Remove noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # White paper + black pen
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV +
        cv2.THRESH_OTSU
    )

    # Find digit
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        contour = max(
            contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        padding = int(
            max(w, h) * 0.25
        )

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

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


    # Make square
    h, w = cropped.shape

    size = max(h, w)

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


    # Resize to MNIST
    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )


    # Normalize
    normalized = (
        resized.astype("float32") / 255.0
    )

    return normalized.reshape(
        1, 28, 28
    )


# ==========================================
# AGENT TOOL — IMAGE QUALITY
# ==========================================

def check_image_quality(image):

    image_array = np.array(image)

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    if np.std(gray) < 10:

        return False

    return True


# ==========================================
# AGENT TOOL — PREDICTION
# ==========================================

def predict_digit(image):

    processed = preprocess_digit_image(
        image
    )

    probabilities = model.predict(
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


# ==========================================
# AGENT
# ==========================================

def digit_agent(image):

    # Check image
    if not check_image_quality(image):

        return {
            "status": "failed",
            "message": "Image is unclear or blank."
        }


    # First prediction
    digit, confidence = predict_digit(
        image
    )


    # Agent decision
    if confidence >= 0.90:

        decision = (
            "High confidence. "
            "Prediction accepted."
        )

    else:

        decision = (
            "Low confidence. "
            "Prediction should be reviewed."
        )


    return {
        "status": "success",
        "digit": digit,
        "confidence": confidence,
        "decision": decision
    }


# ==========================================
# STREAMLIT UI
# ==========================================

st.set_page_config(
    page_title="Handwritten Digit AI",
    page_icon="🤖"
)


st.title(
    "🤖 Agentic AI Handwritten Digit Recognition"
)

st.write(
    "Upload a photo containing one handwritten digit."
)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        width=350
    )


    if st.button(
        "🤖 Predict Digit"
    ):

        with st.spinner(
            "AI Agent is analyzing..."
        ):

            result = digit_agent(
                image
            )


        if result["status"] == "success":

            st.success(
                f"Predicted Digit: "
                f"{result['digit']}"
            )

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.info(
                f"Agent Decision: "
                f"{result['decision']}"
            )

        else:

            st.error(
                result["message"]
            )
