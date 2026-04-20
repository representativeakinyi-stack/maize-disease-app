from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Class labels (must match training order)
classes = ["Healthy", "Blight", "Common Rust", "Gray Leaf Spot"]

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')

    if file is None or file.filename == '':
        return render_template("index.html", prediction="No image uploaded")

    # Open image
    img = Image.open(file).convert("RGB")

    # IMPORTANT: must match training size
    img = img.resize((128, 128))

    # Convert to array and normalize
    img = np.array(img) / 255.0

    # Expand dims for model
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    result = np.argmax(prediction)

    return render_template("index.html", prediction=classes[result])

if __name__ == "__main__":
    app.run(debug=True)