from typing import Optional, Any
from src.ml.models.prediction import CPNSPredictor


class PredictionService:
    _instance = None
    predictor: Optional[CPNSPredictor] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self.predictor:
            self.predictor = CPNSPredictor()

    def predict(self, data: Any) -> dict:
        """
        Make a prediction using the CPNS predictor

        Args:
            data: Input data for prediction

        Returns:
            dict: Prediction results

        Raises:
            RuntimeError: If predictor is not initialized
        """
        if not self.predictor:
            raise RuntimeError("Predictor not initialized")

        try:
            return self.predictor.predict(data)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}") from e

    def train(self, file_path: str) -> bool:
        """
        Train the CPNS predictor model

        Args:
            file_path: Path to training data file

        Returns:
            bool: True if training successful

        Raises:
            RuntimeError: If training fails
        """
        if not self.predictor:
            raise RuntimeError("Predictor not initialized")

        try:
            return self.predictor.train(file_path=file_path, force_retrain=True)
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}") from e
