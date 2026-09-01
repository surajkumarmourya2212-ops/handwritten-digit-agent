import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


# ============================================================
# PAGE SETTINGS
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

st.write(
    "Upload an image containing one handwritten digit (0–9)."
)


# ============================================================
# LOAD YOUR MNIST MODEL
# ============================================================

@st.cache_resource
def load_digit_model():
    return tf.keras.models.load_model("digit_model.keras")


try:
    digit_model = load_digit_model()

except Exception as e:
    st.error("❌ Could not load the digit model.")
    st.code(str(e))
    st.stop()


# ============================================================
# VALIDATE IMAGE
# ============================================================

def is_valid_digit_image(image):

    # Convert image to grayscale
    gray = np.array(image.convert("L"))

    # Resize for validation
    gray = cv2.resize(gray, (200, 200))

    # Blur small noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Convert to black/white
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Keep only meaningful contours
    contours = [
        c for c in contours
        if cv2.contourArea(c) > 30
    ]

    # No digit/object found
    if len(contours) == 0:
        return False

    # Too many separate objects
    if len(contours) > 5:
        return False

    # Largest contour
    largest = max(
        contours,
        key=cv2.contourArea
    )

    largest_area = cv2.contourArea(largest)

    # Total contour area
    total_area = sum(
        cv2.contourArea(c)
        for c in contours
    )

    if total_area == 0:
        return False

    # Largest object should be dominant
    dominance = largest_area / total_area

    if dominance < 0.60:
        return False

    # Bounding box
    x, y, w, h = cv2.boundingRect(largest)

    # Check size
    image_area = binary.shape[0] * binary.shape[1]
    box_area = w * h

    size_ratio = box_area / image_area

    # Too small
    if size_ratio < 0.005:
        return False

    # Too large
    if size_ratio > 0.75:
        return False

    # Check shape
    aspect_ratio = w / float(h)

    if aspect_ratio < 0.15 or aspect_ratio > 5.0:
        return False

    return True


# ============================================================
# PREPROCESS IMAGE FOR MNIST
# ============================================================

def preprocess_digit_image(image):

    # Convert to grayscale
    gray = np.array(
        image.convert("L")
    )

    # Remove small noise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Convert to MNIST style:
    # black background + white digit
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

    # Keep useful contours
    useful = [
        c for c in contours
        if cv2.contourArea(c) > 10
    ]

    # Crop around the main digit
    if useful:

        largest = max(
            useful,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            largest
        )

        # Padding
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

        binary = binary[
            y1:y2,
            x1:x2
        ]

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
    # RESIZE TO 28 × 28
    # ========================================================

    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    resized = (
        resized.astype("float32") / 255.0
    )

    # Model expects (1, 28, 28)
    return resized.reshape(
        1,
        28,
        28
    )


# ============================================================
# UPLOAD IMAGE
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

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Show image
    st.subheader("📷 Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )

    # Predict button
    if st.button(
        "🔍 Predict Digit",
        type="primary"
    ):

        # ----------------------------------------------------
        # STEP 1: VALIDATE IMAGE
        # ----------------------------------------------------

        valid = is_valid_digit_image(
            image
        )

        if not valid:

            st.error(
                "❌ Invalid Image"
            )

            st.warning(
                "Please upload an image containing "
                "ONE handwritten digit (0–9)."
            )

        else:

            # ------------------------------------------------
            # STEP 2: PREPROCESS
            # ------------------------------------------------

            processed = preprocess_digit_image(
                image
            )

            # ------------------------------------------------
            # STEP 3: MODEL PREDICTION
            # ------------------------------------------------

            prediction_output = digit_model.predict(
                processed,
                verbose=0
            )[0]

            # Find highest probability class
            prediction = int(
                np.argmax(prediction_output)
            )

            # ------------------------------------------------
            # STEP 4: SHOW ONLY DIGIT
            # ------------------------------------------------

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
