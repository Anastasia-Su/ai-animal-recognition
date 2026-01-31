from flask import Blueprint, render_template, abort, request
from ml.predictions import predict_image, CLASSES
from ml.utils import save_uploaded_file


router = Blueprint("main", __name__, url_prefix="")


@router.get("/")
def show_form():
    """GET: Render upload form."""

    return render_template("index.html", animal_list=CLASSES)


@router.post("/")
def handle_upload():
    """POST: Handle file upload and return prediction."""

    if "file" not in request.files:
        abort(400, description="No file provided")

    file = request.files["file"]

    try:
        save_path, filename = save_uploaded_file(file)
        predictions, top_class, top_confidence = predict_image(save_path)
    except ValueError as ve:
        abort(400, description=str(ve))
    except Exception as e:
        abort(500, description=str(e))

    return render_template(
        "result.html",
        prediction=predictions,
        top_class=top_class,
        top_confidence=top_confidence,
        image_file=filename,
    )
