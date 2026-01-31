from flask_restx import Api, Namespace, Resource, fields
from flask import request
from ml.utils import save_uploaded_file
from ml.predictions import predict_image


def setup_api(app):
    api = Api(
        app,
        version="1.0",
        title="Animal Classifier API",
        description="Upload image and get prediction",
        doc="/docs",
    )

    ns = Namespace("predict", description="Prediction operations")
    api.add_namespace(ns)

    upload_parser = ns.parser()
    upload_parser.add_argument(
        "file", location="files", type="FileStorage", required=True, help="Image file"
    )

    prediction_model = ns.model(
        "PredictionResponse",
        {
            "top_class": fields.String,
            "top_confidence": fields.Float,
        },
    )

    @ns.route("/")
    class Predict(Resource):

        @ns.expect(upload_parser)
        @ns.marshal_with(prediction_model)
        def post(self):
            """
            Upload image and get prediction
            """

            file = request.files.get("file")

            save_path, filename = save_uploaded_file(file)
            predictions, top_class, top_confidence = predict_image(save_path)

            return {
                "top_class": top_class,
                "top_confidence": top_confidence,
            }

    return api
