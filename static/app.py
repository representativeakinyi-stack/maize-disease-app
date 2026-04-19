from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load trained model
model = load_model("maize_disease_model.h5")

# Class names (must match training order)
classes = ['common_rust', 'gray_leaf_spot', 'healthy', 'northern_leaf_blight']

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    filepath = os.path.join('static/uploads', file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]

    return render_template('index.html', prediction=result, image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)