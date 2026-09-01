import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔢 Handwritten Digit Recognition")
st.write("Upload an image containing one handwritten digit (0–9).")


# ============================================================
# LOAD YOUR MNIST MODEL
# ============================================================

@st.cache_resource
def load_digit_model():
    return tf.keras.models.load_model("digit_model (1).keras")


try:
    digit_model = load_digit_model()
except Exception as e:
    st.error("❌ Could not load the digit model.")
    st.code(str(e))
    st.stop()


# ============================================================
# PREPROCESS IMAGE FOR MNIST MODEL
# Model input: 28 × 28 grayscale image
# ============================================================

def preprocess_digit_image(image):

    # Convert image to grayscale
    image_array = np.array(image.convert("L"))

    # Slight blur to remove image noise
    image_array = cv2.GaussianBlur(
        image_array,
        (3, 3),
        0
    )

    # Convert to MNIST-style:
    # White digit on black background
    _, binary = cv2.threshold(
        image_array,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find the main handwritten digit
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        # Ignore very small noise
        useful = [
            c for c in contours
            if cv2.contourArea(c) > 10
        ]

        if useful:

            # Get largest contour
            contour = max(
                useful,
                key=cv2.contourArea
            )

            x, y, w, h = cv2.boundingRect(contour)

            # Add padding around digit
            padding = int(max(w, h) * 0.25)

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

            binary = binary[y1:y2, x1:x2]

    # ========================================================
    # MAKE IMAGE SQUARE
    # ========================================================

    h, w = binary.shape
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
    ] = binary

    # ========================================================
    # RESIZE TO MNIST SIZE: 28 × 28
    # ========================================================

    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # Normalize pixels from 0–255 to 0–1
    normalized = resized.astype("float32") / 255.0

    # Model expects:
    # (batch, 28, 28)
    return normalized.reshape(
        1,
        28,
        28
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload your handwritten digit image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file:

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Show uploaded image
    st.subheader("📷 Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )

    # Prediction button
    if st.button(
        "🔍 Predict Digit",
        type="primary"
    ):

        # Preprocess image
        processed = preprocess_digit_image(
            image
        )

        # Make prediction
        probabilities = digit_model.predict(
            processed,
            verbose=0
        )[0]

        # Get digit with highest prediction
        prediction = int(
            np.argmax(probabilities)
        )

        # Show only predicted digit
        st.success(
            f"✅ Predicted Digit: {prediction}"
        )


else:

    st.info(
        "Please upload an image of one handwritten digit."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Powered by your trained MNIST handwritten digit model."
        )
