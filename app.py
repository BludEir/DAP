from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = os.getenv('MODEL_PATH', 'static/Model_CNN.h5')
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Define class labels
CLASS_NAMES = ["AD", "MCI", "CN"]

def preprocess_image(image):
    """Preprocesses the uploaded image for model prediction."""
    image = image.convert("L")  # Convert image to grayscale
    image = image.resize((128, 128))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=-1)  # Ensure it has 1 channel
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def home():
    return "Welcome to the Alzheimer's Detection API!"

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image uploads and returns predictions."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess_image(image)

        prediction = model.predict(image)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        app.logger.error(f"Error processing the image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
