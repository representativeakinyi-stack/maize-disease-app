from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid
import gdown

app = Flask(__name__)

os.makedirs("static", exist_ok=True)

MODEL_PATH = "model.keras"
MODEL_URL = "https://drive.google.com/uc?id=11CCmoqktYGezTq46o10nzgwInF3s4FQ5"

print("\n========== MODEL INITIALIZATION ==========")
print("Files:", os.listdir("."))

model = None

try:
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    print("📦 Model size:", os.path.getsize(MODEL_PATH), "bytes")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("✅ MODEL LOADED SUCCESSFULLY")

except Exception as e:
    print("❌ MODEL LOAD ERROR:", str(e))
    model = None


classes = [
    "common_rust",
    "gray_leaf_spot",
    "healthy",
    "northern_leaf_blight"
]


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return "Model failed to load on server"

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join("static", filename)
    file.save(filepath)

    try:
        img = Image.open(filepath).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)