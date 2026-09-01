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
    return tf.keras.models.load_model("models/digit_model.keras")


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
    image_array = np.array(image.convert("L"))

    # Slight blur to remove camera/image noise
    image_array = cv2.GaussianBlur(image_array, (3, 3), 0)

    # MNIST-style: white digit on black background
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
        useful = [c for c in contours if cv2.contourArea(c) > 10]

        if useful:
            contour = max(useful, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)

            # Add padding around the digit
            padding = int(max(w, h) * 0.25)

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(binary.shape[1], x + w + padding)
            y2 = min(binary.shape[0], y + h + padding)

            binary = binary[y1:y2, x1:x2]

    # Make the digit image square
    h, w = binary.shape
    size = max(h, w)

    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = binary

    # Resize to exactly the MNIST model input size
    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # Normalize exactly to 0–1
    normalized = resized.astype("float32") / 255.0

    # Model expects shape: (batch, 28, 28)
    return normalized.reshape(1, 28, 28)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload your handwritten digit image",
    type=["png", "jpg", "jpeg", "webp"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📷 Uploaded Image")
    st.image(image, use_container_width=True)

    if st.button("🔍 Predict Digit", type="primary"):

        processed = preprocess_digit_image(image)

        # Predict
        probabilities = digit_model.predict(
            processed,
            verbose=0
        )[0]

        prediction = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities)) * 100

        st.success(f"✅ Predicted Digit: {prediction}")
        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        # Show all class probabilities
        st.subheader("📊 Prediction Probabilities")

        for digit, probability in enumerate(probabilities):
            st.write(
                f"**{digit}** — {probability * 100:.2f}%"
            )

else:
    st.info("Please upload an image of one handwritten digit.")

st.divider()
st.caption("Powered by your trained MNIST handwritten digit model.")
