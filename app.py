from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

model = load_model("model/animals.keras")

with open("model/class_names.json", "r") as f:
    classes = json.load(f)

# def predict_image(img_path):
#     img = image.load_img(img_path, target_size=(160, 160))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0
#     predictions = model.predict(img_array)
#     max_index = np.argmax(predictions)
#     predicted_class = classes[np.argmax(predictions)]
#     confidence = float(predictions[0][max_index])
#     return predicted_class, confidence


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)[0]

    # Get the top class and its confidence
    top_index = np.argmax(predictions)
    top_class = classes[top_index]
    top_confidence = predictions[top_index]

    # If confidence is less than 60%, show top 3 predictions
    if top_confidence < 0.6:
        top_indices = predictions.argsort()[-3:][::-1]  # Get the top 3 indices
        top_classes = [(classes[i], float(predictions[i])) for i in top_indices]
        return top_classes, top_class, top_confidence
    else:
        return (
            [(top_class, top_confidence)],
            top_class,
            top_confidence,
        )  # Single prediction


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            result, top_class, top_confidence = predict_image(path)
            return render_template(
                "result.html",
                prediction=result,
                top_class=top_class,
                top_confidence=top_confidence,
                image_file=filename,
            )
    return render_template("index.html")


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
