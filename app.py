from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid

app = Flask(__name__)

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# Load trained model
model = tf.keras.models.load_model("model.h5", compile=False)

# Class labels (MUST match training order)
classes = [
    "common_rust",
    "gray_leaf_spot",
    "healthy",
    "northern_leaf_blight"
]

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    # Create unique filename (prevents overwriting)
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join("static", filename)

    file.save(filepath)

    # Load and preprocess image
    img = Image.open(filepath).convert('RGB')
    img = img.resize((128, 128))

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
if __name__ == "__main__":
    print("APP STARTING...")
    app.run(host="0.0.0.0", port=5000)
    # model = tf.keras.models.load_model("model.h5", compile=False)
model = None
if model is None:
    return "Model not loaded"