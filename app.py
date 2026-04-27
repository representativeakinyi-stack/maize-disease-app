from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid

app = Flask(__name__)

# -------------------------
# CREATE STATIC FOLDER
# -------------------------
os.makedirs("static", exist_ok=True)

# -------------------------
# MODEL PATH
# -------------------------
MODEL_PATH = "model.h5"

# -------------------------
# LOAD MODEL AT STARTUP
# -------------------------
print("\n========== MODEL INITIALIZATION ==========")
print("Checking deployed files...")
print(os.listdir("."))

model = None

if os.path.exists(MODEL_PATH):
    try:
        print("Model file found.")
        print("Model size:", os.path.getsize(MODEL_PATH), "bytes")

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        print("✅ MODEL LOADED SUCCESSFULLY")

    except Exception as e:
        print("❌ MODEL LOAD ERROR:", str(e))
        model = None
else:
    print("❌ MODEL FILE NOT FOUND IN DEPLOYMENT")


# -------------------------
# CLASS LABELS
# -------------------------
classes = [
    "common_rust",
    "gray_leaf_spot",
    "healthy",
    "northern_leaf_blight"
]


# -------------------------
# HOME ROUTE
# -------------------------
@app.route('/')
def home():
    return render_template("index.html")


# -------------------------
# PREDICT ROUTE
# -------------------------
@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return "Model failed to load on server"

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    # Save image
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join("static", filename)
    file.save(filepath)

    try:
        # Preprocess image
        img = Image.open(filepath).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)

        predicted_class = classes[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)

        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=f"{confidence:.2f}",
            image=filepath
        )

    except Exception as e:
        print("❌ PREDICTION ERROR:", str(e))
        return "Prediction error"


# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)