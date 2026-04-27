from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid

app = Flask(__name__)

# Create static folder if not exists
os.makedirs("static", exist_ok=True)

# -----------------------
# MODEL CONFIG
# -----------------------
MODEL_PATH = "model_fixed.h5"
model = None


# -----------------------
# LOAD MODEL (DEBUG VERSION)
# -----------------------
def load_model():
    global model

    if model is None:
        try:
            print("===================================")
            print("📁 CHECKING FILES IN CONTAINER:")
            print(os.listdir("."))
            print("===================================")

            if not os.path.exists(MODEL_PATH):
                print("❌ MODEL FILE NOT FOUND:", MODEL_PATH)
                return None

            size = os.path.getsize(MODEL_PATH)
            print(f"📦 MODEL FOUND | SIZE: {size} bytes")

            print("Loading model...")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)

            print("✅ MODEL LOADED SUCCESSFULLY")

        except Exception as e:
            print("❌ MODEL LOAD ERROR:", str(e))
            model = None

    return model


# -----------------------
# CLASSES
# -----------------------
classes = [
    "common_rust",
    "gray_leaf_spot",
    "healthy",
    "northern_leaf_blight"
]


# -----------------------
# HOME ROUTE
# -----------------------
@app.route('/')
def home():
    return render_template("index.html")


# -----------------------
# PREDICT ROUTE
# -----------------------
@app.route('/predict', methods=['POST'])
def predict():
    model = load_model()

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
        img = Image.open(filepath).convert('RGB')
        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
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
        return "Error during prediction"


# -----------------------
# START APP
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)