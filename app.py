from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid

app = Flask(__name__)

os.makedirs("static", exist_ok=True)

# -----------------------
# MODEL PATH
# -----------------------
MODEL_PATH = "model.h5"
model = None


def load_model():
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,
                custom_objects={}
            )
            print("MODEL LOADED SUCCESSFULLY")
        except Exception as e:
            print("MODEL LOAD ERROR:", e)
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

    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join("static", filename)
    file.save(filepath)

    img = Image.open(filepath).convert('RGB')
    img = img.resize((224, 224))   # IMPORTANT: your model uses 224x224

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


# -----------------------
# ENTRY POINT
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)