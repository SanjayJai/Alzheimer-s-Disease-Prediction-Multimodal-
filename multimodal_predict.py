import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing import image

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

def predict_multimodal(img_path, age, gender, mmse):

    # --- Image Prediction ---
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    image_probs = image_model.predict(img_array)[0]

    # --- Clinical Prediction ---
    clinical_input = np.array([[age, gender, mmse]])
    clinical_pred = clinical_model.predict_proba(clinical_input)[0]

    # --- Fusion (Average) ---
    final_probs = (image_probs + clinical_pred) / 2

    final_class = np.argmax(final_probs)

    return class_names[final_class], final_probs


# Example test
if __name__ == "__main__":
    result, probs = predict_multimodal(
        "data/mri_images/NonDemented/1 (53).jpg",  # <-- change to real image
        age=70,
        gender=1,
        mmse=24
    )

    print("Final Prediction:", result)
    print("Probabilities:", probs)