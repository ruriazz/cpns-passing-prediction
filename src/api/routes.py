from flask import Blueprint, request, jsonify, Response
from src.services.prediction_service import PredictionService
from src.api.validators.prediction_request import PredictionRequestSchema
from src.core.config import Config
import os
import uuid

api = Blueprint("api", __name__)


def validate_excel_file(file) -> tuple[str | None, str | None]:
    """Validate uploaded Excel file and return (error, filename) tuple."""
    if not file or file.filename == "":
        return "No file selected", None
    if not file.filename.endswith(".xlsx"):
        return "Invalid file format. Please upload an xlsx file", None
    return None, f"{uuid.uuid4()}.xlsx"


@api.route("/predict", methods=["POST"])
def predict() -> tuple[Response, int]:
    data = PredictionRequestSchema.validate_request(request.get_json())

    if errors := data.get("errors"):
        return jsonify({"success": False, "errors": errors}), 400

    try:
        result = PredictionService().predict(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"success": False, "errors": str(e)}), 500


@api.route("/train", methods=["POST"])
def train() -> tuple[Response, int]:
    file = request.files.get("file")
    error, unique_filename = validate_excel_file(file)

    if error:
        return jsonify({"success": False, "errors": error}), 400

    save_path = os.path.join(Config.BASE_DIR, ".tmp", unique_filename)

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)

        PredictionService().train(file_path=save_path)

        return (
            jsonify({"success": True, "message": "Training completed successfully"}),
            200,
        )
    except Exception as e:
        return jsonify({"success": False, "errors": str(e)}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
