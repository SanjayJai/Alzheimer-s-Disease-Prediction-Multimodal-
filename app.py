from flask import Flask, render_template, request

import os
import numpy as np
import tensorflow as tf
import joblib
from keras.preprocessing import image
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

# Load models
image_model = tf.keras.models.load_model("models/image_model.h5")
clinical_model = joblib.load("models/clinical_model.pkl")

IMG_SIZE = 128


class_names = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

# Example symptoms for each class
class_symptoms = {
    "NonDemented": "No significant memory loss or cognitive decline.",
    "VeryMildDemented": "Very mild memory loss, slight difficulty in complex tasks.",
    "MildDemented": "Noticeable memory loss, confusion, trouble with daily tasks.",
    "ModerateDemented": "Significant memory loss, disorientation, needs assistance with daily activities."
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['mri_image']
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    mmse = int(request.form['mmse'])

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Image prediction
    img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    image_probs = image_model.predict(img_array)[0]

    # Clinical prediction
    clinical_input = np.array([[age, gender, mmse]])
    clinical_probs = clinical_model.predict_proba(clinical_input)[0]

    # Fusion
    final_probs = (image_probs + clinical_probs) / 2
    final_class = np.argmax(final_probs)

    # --- Future confidence increase logic ---
    # Simulate: if today > 6 months from a fixed date, increase confidence by 20%
    base_date = datetime(2026, 3, 4)  # project start date
    now = datetime.now()
    months_passed = (now.year - base_date.year) * 12 + (now.month - base_date.month)
    confidence = float(final_probs[final_class])
    if months_passed >= 6:
        confidence = min(confidence * 1.2, 1.0)
    confidence = round(confidence * 100, 2)

    # Prepare detailed results for all classes
    details = []
    for i, name in enumerate(class_names):
        details.append({
            'name': name,
            'prob': round(float(final_probs[i]) * 100, 2),
            'symptoms': class_symptoms[name]
        })

    result = class_names[final_class]

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        details=details
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)