from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid
import gdown

app = Flask(__name__)

# ----------------------------
# FOLDERS
# ----------------------------
os.makedirs("static", exist_ok=True)

# ----------------------------
# MODEL CONFIG (GOOGLE DRIVE)
# ----------------------------
MODEL_PATH = "model.keras"
MODEL_URL = "https://drive.google.com/uc?id=102FGnGGtTzK0fnVhP-sPSlWWckeuS1VV"

model = None

def load_model():
    global model

    if model is None:
        # Download model if not found
        if not os.path.exists(MODEL_PATH):
            print("Downloading model from Google Drive...")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("MODEL LOADED SUCCESSFULLY")

    return model

# ----------------------------
# CLASSES
# ----------------------------
classes = [
    "common_rust",
    "gray_leaf_spot",
    "healthy",
    "northern_leaf_blight"
]

# ----------------------------
# HOME ROUTE
# ----------------------------
@app.route('/')
def home():
    return render_template("index.html")

# ----------------------------
# PREDICT ROUTE
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    model = load_model()

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join("static", filename)
    file.save(filepath)

    # Preprocess image
    img = Image.open(filepath).convert('RGB')
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

# ----------------------------
# RUN APP (LOCAL ONLY)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)