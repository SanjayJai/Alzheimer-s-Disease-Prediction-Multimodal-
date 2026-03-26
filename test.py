import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load trained image model
model = tf.keras.models.load_model("models/image_model.h5")

IMG_SIZE = 128

class_names = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    probs = model.predict(img_array)[0]
    predicted_class = np.argmax(probs)

    return class_names[predicted_class], probs


# Example Test
if __name__ == "__main__":
    result, probabilities = predict_image(
        "data/mri_images/NonDemented/1 (53).jpg"
    )

    print("Prediction:", result)
    print("Probabilities:", probabilities)