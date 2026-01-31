import os
from flask import Flask
from api_docs import setup_api
from ml.constants import UPLOAD_FOLDER
from ml.routes import router


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESTX_MASK_SWAGGER"] = False

app.register_blueprint(router)

setup_api(app)
# api = Api(
#     app,
#     version="1.0",
#     title="Animal Classifier API",
#     doc="/docs"
# )
# # Define a namespace
# ns = api.namespace("predict", description="ML prediction operations")


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, use_reloader=True)
