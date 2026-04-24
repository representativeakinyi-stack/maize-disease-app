from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid

app = Flask(__name__)

os.makedirs("static", exist_ok=True)

# DO NOT load model at startup (prevents Railway crash)
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("model.h5", compile=False)
    return model


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
    model = load_model()  # safe lazy loading

    file = request.files['file']

    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join("static", filename)
    file.save(filepath)

    img = Image.open(filepath).convert('RGB')
    img = img.resize((128, 128))

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


# Railway / Gunicorn entry point (SAFE)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)