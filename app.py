from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

from tensorflow.keras.models import load_model
model = load_model("model.h5")

# Classes (match your dataset folders)
classes = ['blight', 'common_rust', 'gray_leaf_spot', 'healthy']

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        file = request.files['file']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process image
            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            pred = model.predict(img_array)
            result = classes[np.argmax(pred)]

            prediction = result

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)