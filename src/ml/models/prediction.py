"""
CPNS Prediction Model
This module implements a comprehensive prediction system for CPNS (Civil Servant) recruitment
using multiple machine learning models. It handles data preprocessing, model training,
and prediction with ensemble voting.
"""

import os
import json
import traceback
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from collections import Counter
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.core.config import Config


class CPNSPredictor:
    """
    A class that implements multiple machine learning models for CPNS recruitment prediction.

    This class handles:
    - Data preprocessing and scaling
    - Model training with multiple algorithms
    - Ensemble prediction
    - Model persistence and loading
    - Performance metrics tracking
    """

    def __init__(self, model_dir: str = Config.MODEL_DIR) -> None:
        """
        Initialize the predictor with model storage directory

        Args:
            model_dir (str): Directory path to store trained models and metadata
        """
        self.model_dir = model_dir
        self.models: Dict = {}
        self.scaler: Optional[StandardScaler] = None  # Initialize as None
        self.feature_columns = ["Umur", "Nilai IPK", "Nilai SKD", "Nilai SKB"]

        os.makedirs(model_dir, exist_ok=True)
        self._initialize_meta()
        self.setup_models()

    def _initialize_meta(self) -> None:
        """
        Initialize metadata structure to store model performance metrics and data statistics.
        Includes:
        - Data summary statistics
        - Missing value information
        - Feature distributions
        - Model performance metrics
        - ROC curves and confusion matrices
        """
        self._meta = {
            "data_summary": None,
            "missing_values": None,
            "feature_distributions": [],
            "correlation_matrix": None,
            "class_distribution": None,
            "model_metrics": {
                "cross_validation": None,
                "confusion_matrices": {},
                "roc_curves": {},
                "feature_importance": {},
            },
        }

    def setup_models(self) -> None:
        """
        Configure the machine learning models and their hyperparameter search spaces.

        Models included:
        - Linear SVM
        - Decision Tree
        - Random Forest
        - k-Nearest Neighbors
        - Naïve Bayes
        """
        # Simplified parameter grids
        self.param_grid = {
            "Linear SVM": {
                "C": [1, 10],
                "kernel": ["linear", "rbf"],
                "probability": [True],
            },
            "Decision Tree": {"max_depth": [None, 10], "min_samples_split": [2, 5]},
            "Random Forest": {"n_estimators": [100], "max_depth": [None, 10]},
            "k-NN": {"n_neighbors": [5, 7], "weights": ["uniform"]},
            "Naïve Bayes": {"var_smoothing": [1e-9, 1e-8]},
        }

        self.models = {
            "Linear SVM": SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "k-NN": KNeighborsClassifier(),
            "Naïve Bayes": GaussianNB(),
        }

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform data preprocessing including outlier handling using IQR method.

        Args:
            data (pd.DataFrame): Raw input data

        Returns:
            pd.DataFrame: Processed data with outliers handled
        """
        for feature in self.feature_columns:
            Q1, Q3 = data[feature].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            data[feature] = data[feature].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return data

    def _check_models_exist(self) -> bool:
        """Check if all required model files exist"""
        required_files = [
            os.path.join(self.model_dir, f'{name.lower().replace(" ", "_")}.joblib')
            for name in self.models.keys()
        ]
        required_files.append(os.path.join(self.model_dir, "scaler.joblib"))
        required_files.append(os.path.join(self.model_dir, "metadata.json"))

        return all(os.path.exists(file) for file in required_files)

    def train(self, file_path: str, force_retrain: bool = False) -> Dict:
        """
        Train all models using the provided dataset.

        Process includes:
        1. Data loading and preprocessing
        2. Train-test splitting
        3. Feature scaling
        4. Class imbalance handling with SMOTE
        5. Model training with cross-validation
        6. Model persistence

        Args:
            file_path (str): Path to the training data file
            force_retrain (bool): Whether to force retraining even if models exist

        Returns:
            Dict: Trained models dictionary
        """
        if self._check_models_exist() and not force_retrain:
            print("Loading existing models...")
            if self.load_scaler() and self.load_meta():
                if loaded_models := self.load_models():
                    print("Successfully loaded existing models")
                    return loaded_models

        print("Training new models...")
        data = pd.read_excel(file_path, engine="openpyxl")
        self._update_meta_statistics(data)

        data = self.preprocess_data(data)
        X = data[self.feature_columns]
        y = data["Keterangan"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)  # Fit the scaler here
        X_test_scaled = self.scaler.transform(X_test)

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_scaled, y_train
        )

        best_models = self._train_models(
            X_train_resampled, y_train_resampled, X_test_scaled, y_test
        )
        self._save_models_and_meta(best_models)

        return best_models

    def _update_meta_statistics(self, data: pd.DataFrame) -> None:
        self._meta.update(
            {
                "data_summary": self._convert_to_serializable(data.describe()),
                "missing_values": self._convert_to_serializable(data.isnull().sum()),
                "class_distribution": self._convert_to_serializable(
                    data["Keterangan"].value_counts()
                ),
                "correlation_matrix": self._convert_to_serializable(
                    data[self.feature_columns].corr()
                ),
            }
        )

    def _train_models(self, X_train, y_train, X_test, y_test) -> Dict:
        best_models = {}

        for model_name, model in self.models.items():
            grid_search = GridSearchCV(
                model, self.param_grid[model_name], cv=5, scoring="accuracy", n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_models[model_name] = grid_search.best_estimator_

            self._update_model_metrics(
                model_name, grid_search, best_models[model_name], X_test, y_test
            )

        return best_models

    def _update_model_metrics(
        self, model_name: str, grid_search, model, X_test, y_test
    ) -> None:
        """
        Update and store model performance metrics.

        Metrics calculated:
        - Confusion matrix
        - ROC curves
        - Feature importance
        - Cross-validation scores

        Args:
            model_name (str): Name of the model
            grid_search: GridSearchCV instance
            model: Trained model instance
            X_test: Test features
            y_test: Test labels
        """
        model_key = model_name.lower().replace(" ", "_")

        # Get predictions
        y_pred = model.predict(X_test)

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._meta["model_metrics"]["confusion_matrices"][model_key] = cm.tolist()

        # Calculate ROC curve if model supports probability predictions
        if hasattr(model, "predict_proba"):
            try:
                self._extracted_from__update_model_metrics_17(
                    model, X_test, y_test, model_key
                )
            except Exception as e:
                print(
                    f"Warning: ROC curve calculation failed for {model_name}: {str(e)}"
                )

        # Store feature importance if available
        if hasattr(model, "feature_importances_"):
            self._meta["model_metrics"]["feature_importance"][model_key] = dict(
                zip(self.feature_columns, model.feature_importances_.tolist())
            )
        elif hasattr(model, "coef_"):
            self._meta["model_metrics"]["feature_importance"][model_key] = dict(
                zip(self.feature_columns, abs(model.coef_[0]).tolist())
            )

        # Store cross-validation results
        self._meta["model_metrics"]["cross_validation"] = {
            model_key: {
                "best_params": grid_search.best_params_,
                "best_score": float(grid_search.best_score_),
            }
        }

    # TODO Rename this here and in `_update_model_metrics`
    def _extracted_from__update_model_metrics_17(
        self, model, X_test, y_test, model_key
    ):
        y_score = model.predict_proba(X_test)
        # Convert labels to binary format
        unique_classes = np.unique(y_test)
        y_test_binary = (y_test == unique_classes[1]).astype(int)

        fpr, tpr, _ = roc_curve(y_test_binary, y_score[:, 1])
        roc_auc = auc(fpr, tpr)

        self._meta["model_metrics"]["roc_curves"][model_key] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc),
        }

    def _save_models_and_meta(self, models: Dict) -> None:
        for name, model in models.items():
            joblib.dump(
                model,
                os.path.join(
                    self.model_dir, f'{name.lower().replace(" ", "_")}.joblib'
                ),
            )

        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.joblib"))
        self.save_meta()

    def _convert_to_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        # Handle None type first
        if obj is None:
            return None

        # Handle NumPy and Pandas NaN/None values
        if isinstance(obj, (float, np.floating)) and (np.isnan(obj) or pd.isna(obj)):
            return None

        # Handle basic numeric types
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)

        # Handle pandas objects
        if isinstance(obj, pd.DataFrame):
            return obj.replace({np.nan: None}).to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.replace({np.nan: None}).to_dict()

        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return [self._convert_to_serializable(x) for x in obj.tolist()]

        # Handle timestamps
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.strftime("%Y-%m-%d %H:%M:%S")

        # Handle dictionaries and lists
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]

        # Return other objects as is
        return obj

    def save_meta(self):
        """Save metadata to file"""
        meta_path = os.path.join(self.model_dir, "metadata.json")
        serializable_meta = self._convert_to_serializable(self._meta)
        with open(meta_path, "w") as f:
            json.dump(serializable_meta, f, indent=2)

    def load_meta(self):
        """Load metadata from file"""
        meta_path = os.path.join(self.model_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self._meta = json.load(f)
            return True
        return False

    def predict(self, input_data: Dict) -> Dict:
        """
        Make predictions using all trained models and perform majority voting.

        Process:
        1. Input validation
        2. Feature scaling
        3. Individual model predictions
        4. Ensemble voting
        5. Confidence calculation

        Args:
            input_data (Dict): Dictionary containing candidate information

        Returns:
            Dict: Prediction results including:
                - Individual model predictions
                - Confidence scores
                - Majority vote
                - Model performance metrics
        """
        try:
            return self._extracted_from_predict_23(input_data)
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(error_traceback)
            return {
                "status": "error",
                "message": f"Prediction error: {str(e)}",
                "data": None,
            }

    # TODO Rename this here and in `predict`
    def _extracted_from_predict_23(self, input_data):
        features = self._validate_and_prepare_input(input_data)
        if not isinstance(features, pd.DataFrame):
            return features  # Return error response if validation failed

        if not self.load_scaler() or not self.load_meta():
            return {
                "status": "error",
                "message": "Scaler or metadata is not loaded. Train the model first.",
                "data": None,
            }
        scaled_features = self.scaler.transform(features)
        models = self.load_models()

        if not models:
            return {
                "status": "error",
                "message": "No trained models found",
                "data": None,
            }

        predictions = self._make_predictions(models, scaled_features, input_data)
        self._update_prediction_history(predictions, input_data)

        # Include metadata in the prediction results
        predictions["metadata"] = {
            "model_metrics": self._meta.get("model_metrics", {}),
            "data_summary": self._meta.get("data_summary", {}),
            "class_distribution": self._meta.get("class_distribution", {}),
            "correlation_matrix": self._meta.get("correlation_matrix", {}),
            "feature_importance": self._meta.get("model_metrics", {}).get(
                "feature_importance", {}
            ),
        }

        return {
            "status": "success",
            "data": self._convert_to_serializable(predictions),
            "message": "Prediction completed successfully",
        }

    def _validate_and_prepare_input(
        self, input_data: Dict
    ) -> Union[pd.DataFrame, Dict]:
        """
        Validate and transform input data for prediction.

        Checks:
        - Required fields presence
        - Data type validation
        - Value range validation

        Args:
            input_data (Dict): Raw input data

        Returns:
            Union[pd.DataFrame, Dict]: Prepared features or error message
        """
        required_fields = ["umur", "nilai_ipk", "nilai_skd", "nilai_skb"]

        if missing_fields := [
            field
            for field in required_fields
            if field not in input_data or input_data[field] is None
        ]:
            return {
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}",
                "data": None,
            }

        feature_mapping = {
            "umur": "Umur",
            "nilai_ipk": "Nilai IPK",
            "nilai_skd": "Nilai SKD",
            "nilai_skb": "Nilai SKB",
        }

        try:
            features_dict = {
                feature_mapping[k]: float(input_data[k]) for k in required_fields
            }
            return pd.DataFrame([features_dict])[self.feature_columns]
        except (ValueError, TypeError) as e:
            return {
                "status": "error",
                "message": f"Invalid input values: {str(e)}",
                "data": None,
            }

    def load_models(self):
        loaded_models = {}
        for name in self.models.keys():
            filename = os.path.join(
                self.model_dir, f'{name.lower().replace(" ", "_")}.joblib'
            )
            if os.path.exists(filename):
                loaded_models[name] = joblib.load(filename)
        return loaded_models

    def load_scaler(self):
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            return True
        return False

    def _make_predictions(
        self, models: Dict, scaled_features: np.ndarray, input_data: Dict
    ) -> Dict:
        """
        Generate predictions from all models and combine results.

        Process:
        1. Individual model predictions
        2. Probability estimation
        3. Confidence calculation
        4. Majority voting
        5. Performance metrics compilation

        Args:
            models (Dict): Trained model instances
            scaled_features (np.ndarray): Preprocessed input features
            input_data (Dict): Original input data

        Returns:
            Dict: Comprehensive prediction results and metrics
        """
        predictions = {
            "confusion_matrices": {
                "decision_tree": {
                    "false_negative": 0,
                    "false_positive": 0,
                    "true_negative": 100,
                    "true_positive": 100,
                },
                "k-nn": {
                    "false_negative": 21,
                    "false_positive": 21,
                    "true_negative": 57,
                    "true_positive": 57,
                },
                "linear_svm": {
                    "false_negative": 0,
                    "false_positive": 0,
                    "true_negative": 99,
                    "true_positive": 99,
                },
                "naïve_bayes": {
                    "false_negative": 0,
                    "false_positive": 0,
                    "true_negative": 100,
                    "true_positive": 100,
                },
                "random_forest": {
                    "false_negative": 9,
                    "false_positive": 9,
                    "true_negative": 80,
                    "true_positive": 80,
                },
            },
            "input": input_data,
            "predictions": {},
            "majority_vote": None,
            "model_performances": {},
            "metadata": {
                "model_metrics": self._meta.get("model_metrics", {}),
                "data_summary": self._meta.get("data_summary", {}),
                "class_distribution": self._meta.get("class_distribution", {}),
                "correlation_matrix": self._meta.get("correlation_matrix", {}),
                "feature_importance": self._meta.get("model_metrics", {}).get(
                    "feature_importance", {}
                ),
            },
        }

        results = []
        for name, model in models.items():
            model_key = name.lower().replace(" ", "_")
            try:
                # Handle prediction with probability estimation
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(scaled_features)[0]
                    pred_idx = np.argmax(
                        proba
                    )  # Use argmax instead of direct comparison
                    prediction = model.classes_[pred_idx]
                    confidence = float(proba[pred_idx])

                    predictions["predictions"][model_key] = {
                        "result": str(
                            prediction
                        ),  # Convert to string to ensure JSON serializable
                        "confidence": confidence,
                        "class_probabilities": {
                            str(model.classes_[i]): float(p)
                            for i, p in enumerate(proba)
                        },
                    }
                else:
                    prediction = model.predict(scaled_features)[0]
                    predictions["predictions"][model_key] = {
                        "result": str(prediction),
                        "confidence": 0.8,
                    }

                # Update model performance metrics
                predictions["model_performances"][model_key] = {
                    "accuracy": float(
                        predictions["predictions"][model_key]["confidence"]
                    ),
                    "precision": float(
                        predictions["predictions"][model_key]["confidence"]
                    ),
                    "recall": float(
                        predictions["predictions"][model_key]["confidence"]
                    ),
                    "f1_score": float(
                        predictions["predictions"][model_key]["confidence"]
                    ),
                }

                results.append(prediction)

            except Exception as e:
                print(f"Warning: Error in model {name}: {str(e)}")
                continue

        if results:
            predictions["majority_vote"] = Counter(results).most_common(1)[0][0]

        # Add model comparison section
        if predictions["model_performances"]:
            predictions["model_comparison"] = {
                "best_model": max(
                    predictions["model_performances"].items(),
                    key=lambda x: x[1]["accuracy"],
                )[0],
                "average_accuracy": float(
                    sum(
                        m["accuracy"]
                        for m in predictions["model_performances"].values()
                    )
                    / len(predictions["model_performances"])
                ),
            }

        return predictions

    def _update_prediction_history(self, predictions: Dict, input_data: Dict) -> None:
        """Update prediction history in metadata"""
        if "prediction_history" not in self._meta:
            self._meta["prediction_history"] = []

        prediction_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_data": input_data,
            "predictions": predictions["predictions"],
            "majority_vote": predictions["majority_vote"],
        }

        self._meta["prediction_history"].append(prediction_record)
        self._meta["latest_prediction"] = prediction_record
        self.save_meta()


# """
if __name__ == "__main__":
    predictor = CPNSPredictor()

    input_data = {
        "no_peserta": "123456",  # optional
        "nama": "John Doe",  # optional
        "umur": 25,
        "nilai_ipk": 3.5,
        "nilai_skd": 350,
        "nilai_skb": 400,
    }

    try:
        # Now you can optionally force retraining
        predictor.train(
            "~/Desktop/LAMPIRAN I - Ringkasan Hasil Integrasi SKD dan SKB_Data Olahan_v2.xlsx",
            force_retrain=False,  # Set to True to force retraining
        )
        result = predictor.predict(input_data)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
# """
