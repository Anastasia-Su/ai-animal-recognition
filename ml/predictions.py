import json
import numpy as np
import ml.constants as constants
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model(constants.MODEL_PATH)

with open(constants.CLASSES_PATH, "r") as f:
    CLASSES = json.load(f)


def preprocess_image(img_path: str) -> np.ndarray:
    """Load and preprocess image for model prediction."""
    img = image.load_img(img_path, target_size=constants.IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


def get_top_predictions(
    predictions: np.ndarray,
) -> tuple[list[tuple[str, float]], str, float]:
    """Return prediction results depending on confidence level."""

    top_index = int(np.argmax(predictions))
    top_class = CLASSES[top_index]
    top_confidence = float(predictions[top_index])

    if top_confidence < constants.CONFIDENCE_THRESHOLD:
        top_indices = predictions.argsort()[-constants.TOP_K_LOW_CONF:][::-1]
        results = [(CLASSES[i], float(predictions[i])) for i in top_indices]
    else:
        results = [(top_class, top_confidence)]

    return results, top_class, top_confidence


def predict_image(img_path: str):
    """Full prediction pipeline."""
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img, verbose=0)[0]

    return get_top_predictions(predictions)
