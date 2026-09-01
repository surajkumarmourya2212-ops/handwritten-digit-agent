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

st.write(
    "Upload an image containing one handwritten digit (0–9)."
)


# ============================================================
# LOAD MNIST MODEL
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
# CHECK WHETHER IMAGE LOOKS LIKE A SINGLE DIGIT
# ============================================================

def validate_digit_image(image):

    # Convert to grayscale
    gray = np.array(image.convert("L"))

    # Resize for checking
    check_image = cv2.resize(
        gray,
        (200, 200)
    )

    # Blur image
    blurred = cv2.GaussianBlur(
        check_image,
        (3, 3),
        0
    )

    # Convert black/white
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove small noise
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

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

    # Keep meaningful contours
    useful_contours = [
        c for c in contours
        if cv2.contourArea(c) > 30
    ]

    # No foreground object
    if len(useful_contours) == 0:
        return False

    # Sort contours by area
    useful_contours.sort(
        key=cv2.contourArea,
        reverse=True
    )

    largest_contour = useful_contours[0]

    largest_area = cv2.contourArea(
        largest_contour
    )

    # --------------------------------------------------------
    # Check number of objects
    # --------------------------------------------------------

    # Too many separate objects usually means
    # photo/text/multiple objects
    if len(useful_contours) > 5:
        return False

    # --------------------------------------------------------
    # Check whether largest object dominates
    # --------------------------------------------------------

    total_area = sum(
        cv2.contourArea(c)
        for c in useful_contours
    )

    if total_area == 0:
        return False

    largest_ratio = (
        largest_area / total_area
    )

    if largest_ratio < 0.60:
        return False

    # --------------------------------------------------------
    # Check bounding box
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    image_area = (
        binary.shape[0] *
        binary.shape[1]
    )

    box_area = w * h

    box_ratio = (
        box_area / image_area
    )

    # Object too small
    if box_ratio < 0.005:
        return False

    # Object covers almost entire image
    if box_ratio > 0.75:
        return False

    # --------------------------------------------------------
    # Check aspect ratio
    # --------------------------------------------------------

    aspect_ratio = w / float(h)

    # Extremely wide/tall objects are unlikely
    # to be a single MNIST-style digit
    if aspect_ratio < 0.15 or aspect_ratio > 5.0:
        return False

    return True


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_digit_image(image):

    # Convert to grayscale
    image_array = np.array(
        image.convert("L")
    )

    # Remove small noise
    image_array = cv2.GaussianBlur(
        image_array,
        (3, 3),
        0
    )

    # MNIST style:
    # white digit on black background
    _, binary = cv2.threshold(
        image_array,
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

    if contours:

        useful = [
            c for c in contours
            if cv2.contourArea(c) > 10
        ]

        if useful:

            # Select largest contour
            contour = max(
                useful,
                key=cv2.contourArea
            )

            # Get bounding box
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
                y -
