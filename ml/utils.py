import os
from werkzeug.utils import secure_filename
from ml.constants import UPLOAD_FOLDER


def save_uploaded_file(file):
    """Save uploaded file to upload folder."""
    if file.filename == "":
        raise ValueError("Empty filename")

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    return save_path, filename
