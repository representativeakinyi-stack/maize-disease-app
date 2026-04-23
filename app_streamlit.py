import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# =========================
# LOAD MODEL (MUST BE FIRST)
# =========================
model = load_model("model.h5")

# =========================
# CLASS LABELS (YOUR TRAINING ORDER)
# =========================
class_names = [
    "Common Rust",
    "Gray Leaf Spot",
    "Healthy",
    "Northern Leaf Blight"
]

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Maize Disease Detection", layout="centered")

st.title("🌽 Web-Based Maize Disease Detection System")
st.write("Upload a maize leaf image to detect disease using CNN")

# =========================
# IMAGE UPLOAD
# =========================
file = st.file_uploader("Choose a maize leaf image", type=["jpg", "jpeg", "png"])

if file is not None:

    # Load image
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # =========================
    # PREPROCESSING (MUST MATCH MODEL INPUT: 128x128)
    # =========================
    img = image.resize((128, 128))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(img)[0]

    class_index = np.argmax(prediction)
    confidence = float(prediction[class_index])

    # =========================
    # OUTPUT
    # =========================
    st.subheader("Prediction Result:")
    st.success(class_names[class_index])

    st.write("Confidence:")
    st.info(f"{confidence * 100:.2f}%")

    # =========================
    # DEBUG SECTION (IMPORTANT)
    # =========================
    with st.expander("View Raw Model Output (Debug)"):
        st.write("Prediction probabilities:")
        st.write(prediction)

        st.write("Predicted class index:")
        st.write(class_index)