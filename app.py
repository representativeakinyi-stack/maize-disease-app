from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid
import requests

app = Flask(__name__)

# Create static folder
os.makedirs("static", exist_ok=True)

# -----------------------
# MODEL CONFIG
# -----------------------
MODEL_PATH = "model_fixed.h5"
MODEL_URL = "https://drive.google.com/uc?id=1Y9pKbUfo80mf9PQ1wblLKZONjrldkFCO"

model = None


# -----------------------
# DOWNLOAD FUNCTION
# -----------------------
def download_model():
    print("Downloading model from Google Drive...")

    response = requests.get(MODEL_URL, stream=True)

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Download complete")


# -----------------------
# LOAD MODEL
# -----------------------
def load_model():
    global model

    if model is None:
        try:
            print("Files in directory:", os.listdir("."))

            if not os.path.exists(MODEL_PATH):
                download_model()

            print("Loading model...")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)

            print("MODEL LOADED SUCCESSFULLY")

        except Exception as e:
            print("MODEL LOAD ERROR:", str(e))
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
# ROUTES
# -----------------------
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    model = load_model()

    if model is None:
        return "Model failed to load on server"

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    # Save uploaded image
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
        print("PREDICTION ERROR:", str(e))
        return "Error during prediction"


# -----------------------
# ENTRY POINT
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)