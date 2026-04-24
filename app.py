from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid

app = Flask(__name__)

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# Global model (lazy loaded)
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("model.h5", compile=False)
    return model


# Class labels (must match training order)
classes = [
    "common_rust",
    "gray_leaf_spot",
    "healthy",
    "northern_leaf_blight"
]


# Home page
@app.route('/')
def home():
    return render_template("index.html")


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = load_model()

        # Validate file upload
        if 'file' not in request.files:
            return "No file uploaded"

        file = request.files['file']

        if file.filename == '':
            return "No file selected"

        # Save image safely
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join("static", filename)
        file.save(filepath)

        # Preprocess image
        img = Image.open(filepath).convert('RGB')
        img = img.resize((128, 128))

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
        print("ERROR:", e)
        return f"Prediction error: {str(e)}"


# Railway / Gunicorn entry point
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)