import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Handwritten Digit AI",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.keras")


model = load_model()


# ============================================================
# IMAGE PREPROCESSING TOOL
# ============================================================

def preprocess_digit_image(image):

    # Convert PIL image to NumPy array
    image = np.array(image)

    # Convert RGB to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = image.copy()

    # Remove small noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Detect whether image is light or dark
    # White paper + black pen
    if np.mean(gray) > 127:

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV +
            cv2.THRESH_OTSU
        )

    # Black background + white digit
    else:

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY +
            cv2.THRESH_OTSU
        )

    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cropped = binary

    # Find the handwritten digit
    if contours:

        image_area = binary.shape[0] * binary.shape[1]

        valid_contours = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < image_area * 0.001:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Ignore objects that cover almost the entire image
            if (w * h) < image_area * 0.80:

                valid_contours.append(
                    contour
                )

        if valid_contours:

            contour = max(
                valid_contours,
                key=cv2.contourArea
            )

            x, y, w, h = cv2.boundingRect(
                contour
            )

            # Add padding around digit
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

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = cropped

    # ========================================================
    # RESIZE TO MNIST 28x28
    # ========================================================

    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    normalized = (
        resized.astype("float32") / 255.0
    )

    # Shape required by neural network
    processed = normalized.reshape(
        1,
        28,
        28
    )

    return processed


# ============================================================
# AGENT TOOL 1 — IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(image):

    image_array = np.array(image)

    # Convert to grayscale
    if len(image_array.shape) == 3:

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

    else:

        gray = image_array

    # Check contrast
    contrast = np.std(gray)

    if contrast < 10:

        return (
            False,
            "Image appears blank or unclear."
        )

    return (
        True,
        "Image quality is acceptable."
    )


# ============================================================
# AGENT TOOL 2 — DIGIT PREDICTION
# ============================================================

def predict_digit(image):

    # Preprocess image
    processed = preprocess_digit_image(
        image
    )

    # Get model prediction
    probabilities = model.predict(
        processed,
        verbose=0
    )[0]

    # Find digit with highest probability
    digit = int(
        np.argmax(probabilities)
    )

    # Find confidence
    confidence = float(
        np.max(probabilities)
    )

    return (
        digit,
        confidence,
        probabilities
    )


# ============================================================
# AGENT TOOL 3 — IMAGE ENHANCEMENT
# ============================================================

def enhance_digit_image(image):

    image_array = np.array(image)

    # Convert to grayscale
    if len(image_array.shape) == 3:

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

    else:

        gray = image_array

    # Improve contrast
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Threshold
    _, enhanced = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV +
        cv2.THRESH_OTSU
    )

    return enhanced


# ============================================================
# 🤖 AGENT
# ============================================================

def digit_agent(image):

    # --------------------------------------------------------
    # STEP 1 — CHECK IMAGE
    # --------------------------------------------------------

    valid, quality_message = (
        check_image_quality(image)
    )

    if not valid:

        return {
            "status": "failed",
            "message": quality_message
        }


    # --------------------------------------------------------
    # STEP 2 — FIRST PREDICTION
    # --------------------------------------------------------

    digit1, confidence1, probabilities1 = (
        predict_digit(image)
    )


    # --------------------------------------------------------
    # STEP 3 — AGENT DECISION
    # --------------------------------------------------------

    if confidence1 >= 0.90:

        return {
            "status": "success",
            "digit": digit1,
            "confidence": confidence1,
            "attempts": 1,
            "decision": (
                "High confidence. "
                "Prediction accepted."
            )
        }


    # --------------------------------------------------------
    # STEP 4 — LOW CONFIDENCE
    # AGENT TRIES IMAGE ENHANCEMENT
    # --------------------------------------------------------

    enhanced = enhance_digit_image(
        image
    )

    # Convert enhanced image to PIL
    enhanced_pil = Image.fromarray(
        enhanced
    ).convert("RGB")


    # Second prediction
    digit2, confidence2, probabilities2 = (
        predict_digit(enhanced_pil)
    )


    # --------------------------------------------------------
    # STEP 5 — COMPARE PREDICTIONS
    # --------------------------------------------------------

    if confidence2 > confidence1:

        return {
            "status": "success",
            "digit": digit2,
            "confidence": confidence2,
            "attempts": 2,
            "decision": (
                "Initial confidence was low. "
                "Image was enhanced and the "
                "more confident prediction was selected."
            )
        }

    else:

        return {
            "status": "success",
            "digit": digit1,
            "confidence": confidence1,
            "attempts": 2,
            "decision": (
                "Initial prediction was retained "
                "because it had higher confidence."
            )
        }


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title(
    "🤖 Agentic AI Handwritten Digit Recognition"
)

st.write(
    "Upload a handwritten digit image and "
    "let the AI agent recognize it."
)

st.info(
    "For best results, upload an image containing "
    "one handwritten digit."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload handwritten digit",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ============================================================
# DISPLAY AND PREDICT
# ============================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # Display image
    st.subheader(
        "📷 Uploaded Image"
    )

    st.image(
        image,
        width=350
    )


    # Predict button
    if st.button(
        "🤖 Predict Digit",
        type="primary"
    ):

        with st.spinner(
            "AI Agent is analyzing the image..."
        ):

            result = digit_agent(
                image
            )


        # ====================================================
        # RESULT
        # ====================================================

        if result["status"] == "success":

            st.subheader(
                "🎯 Prediction Result"
            )

            st.success(
                f"Predicted Digit: {result['digit']}"
            )

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.write(
                f"**Number of Attempts:** "
                f"{result['attempts']}"
            )

            st.info(
                f"🤖 **Agent Decision:** "
                f"{result['decision']}"
            )


        else:

            st.error(
                f"❌ {result['message']}"
            )


# ============================================================
# PROJECT INFORMATION
# ============================================================

with st.expander(
    "ℹ️ About this project"
):

    st.write(
        """
        This application uses an Agentic AI layer
        together with a trained MNIST neural network.

        The agent:

        1. Checks the uploaded image.
        2. Preprocesses the image.
        3. Sends the image to the trained model.
        4. Checks the prediction confidence.
        5. Enhances the image when confidence is low.
        6. Compares predictions.
        7. Selects the final prediction.
        """
    )
