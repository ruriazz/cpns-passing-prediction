"""
Optimized CPNS Prediction Model

This module represents an enhanced and optimized version of the CPNS prediction system.
Key improvements include:
- Streamlined data preprocessing
- Enhanced model performance tracking
- Improved error handling and validation
- Better memory management
- More robust prediction confidence calculation

The model uses an ensemble of different machine learning algorithms to predict
CPNS (Civil Servant) recruitment outcomes based on candidate data.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)
from collections import Counter


class CPNSPredictor:
    """
    Enhanced predictor for CPNS recruitment outcomes using ensemble learning.

    This class implements multiple machine learning models working together to provide
    more accurate and reliable predictions. It includes built-in data validation,
    preprocessing, and comprehensive performance tracking.

    Key Features:
    - Automated data cleaning and validation
    - Ensemble voting mechanism
    - Model performance tracking
    - Persistent storage of models and metadata
    - Comprehensive error handling
    """

    def __init__(self, model_dir: str = "./.trained") -> None:
        """
        Initialize the predictor with necessary components.

        Args:
            model_dir: Directory path for storing trained models and metadata.
                      Defaults to './.trained'
        """
        self.model_dir = model_dir
        self.models = {}
        self.feature_columns = ["Umur", "Nilai IPK", "Nilai SKD", "Nilai SKB"]
        self.target_column = "Keterangan"
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        os.makedirs(model_dir, exist_ok=True)
        self._initialize_meta()
        self._setup_models()

    def _initialize_meta(self) -> None:
        """
        Set up the metadata structure for tracking model performance and data statistics.

        The metadata includes:
        - Statistical summaries of training data
        - Missing value information
        - Feature distributions and correlations
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
                "confusion_matrices": {},
                "roc_curves": {},
                "feature_importance": {},
            },
        }

    def _setup_models(self) -> None:
        """
        Configure the ensemble of machine learning models.

        Includes:
        - Support Vector Machine (SVM) for complex decision boundaries
        - Decision Tree for interpretable predictions
        - Random Forest for robust ensemble predictions
        - K-Nearest Neighbors for similarity-based classification
        - Naive Bayes for probability-based predictions
        """
        self.models = {
            "SVM": SVC(kernel="linear", probability=True, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "K-NN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
        }

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning and preprocessing.

        Steps:
        1. Handle date formats and missing dates
        2. Process numeric features
        3. Remove outliers using IQR method
        4. Handle missing values with appropriate strategies
        5. Remove duplicates and incomplete records

        Args:
            data: Raw input DataFrame

        Returns:
            Cleaned and preprocessed DataFrame
        """
        clean_data = data.copy()

        # Handle date columns first
        if "Tanggal Lahir" in clean_data.columns:
            clean_data.loc[:, "Tanggal Lahir"] = pd.to_datetime(
                clean_data["Tanggal Lahir"], errors="coerce"
            )
            # Fill missing dates with median date
            median_date = clean_data["Tanggal Lahir"].dropna().median()
            clean_data.loc[clean_data["Tanggal Lahir"].isnull(), "Tanggal Lahir"] = (
                median_date
            )

        # Handle numeric features
        for feature in self.feature_columns:
            if clean_data[feature].isnull().any():
                median_value = clean_data[feature].median()
                clean_data.loc[clean_data[feature].isnull(), feature] = median_value

            Q1 = clean_data[feature].quantile(0.25)
            Q3 = clean_data[feature].quantile(0.75)
            IQR = Q3 - Q1
            clean_data.loc[:, feature] = np.clip(
                clean_data[feature], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            )

        # Remove duplicates and remaining null values
        clean_data = clean_data.drop_duplicates()
        return clean_data.dropna(subset=self.feature_columns + [self.target_column])

    def train(self, file_path: str, force_retrain: bool = False) -> dict:
        """
        Train the ensemble of models on provided data.

        Process:
        1. Data loading and cleaning
        2. Feature scaling and encoding
        3. Class imbalance handling
        4. Model training and validation
        5. Performance metrics calculation
        6. Model persistence

        Args:
            file_path: Path to training data file
            force_retrain: Whether to retrain existing models

        Returns:
            Dictionary of trained models
        """
        if (
            self._check_models_exist()
            and not force_retrain
            and (self.load_models() and self.load_scaler() and self.load_meta())
        ):
            return self.models

        data = pd.read_excel(file_path)
        data = self._clean_data(data)
        self._update_meta_statistics(data)

        X = data[self.feature_columns]
        y = self.label_encoder.fit_transform(data[self.target_column])

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        trained_models = {}
        for model_name, model in self.models.items():
            model.fit(X_train_resampled, y_train_resampled)
            trained_models[model_name] = model

            self._update_model_metrics(model_name, model, X_test, y_test)

        self.models = trained_models
        self._save_models_and_meta()

        return trained_models

    def predict(self, input_data: dict) -> dict:
        """
        Predict CPNS recruitment outcomes based on input data.

        Args:
            input_data: Dictionary containing input features

        Returns:
            Dictionary containing prediction results and metadata
        """
        try:
            features = self._validate_and_prepare_input(input_data)
            if isinstance(features, dict) and features.get("status") == "error":
                return features

            scaled_features = self.scaler.transform(features)
            predictions = self._make_predictions(
                self.models, scaled_features, input_data
            )
            self._update_prediction_history(predictions, input_data)

            return {
                "status": "success",
                "data": self._convert_to_serializable(predictions),
                "message": "Prediction completed successfully",
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Prediction error: {str(e)}",
                "data": None,
            }

    def _update_model_metrics(self, model_name: str, model, X_test, y_test) -> None:
        """
        Update performance metrics for a given model.

        Metrics include:
        - Confusion matrix
        - ROC curve and AUC
        - Feature importance

        Args:
            model_name: Name of the model
            model: Trained model instance
            X_test: Test features
            y_test: Test labels
        """
        model_key = model_name.lower().replace(" ", "_")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Update confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._meta["model_metrics"]["confusion_matrices"][model_key] = cm.tolist()

        # Update ROC curves
        if hasattr(model, "predict_proba"):
            y_test_bin = pd.get_dummies(y_test)
            fpr, tpr, _ = roc_curve(y_test_bin.values.ravel(), y_pred_proba.ravel())
            roc_auc = auc(fpr, tpr)
            self._meta["model_metrics"]["roc_curves"][model_key] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc),
            }

        # Update feature importance
        if hasattr(model, "feature_importances_"):
            self._meta["model_metrics"]["feature_importance"][model_key] = dict(
                zip(self.feature_columns, model.feature_importances_.tolist())
            )
        elif hasattr(model, "coef_"):
            self._meta["model_metrics"]["feature_importance"][model_key] = dict(
                zip(self.feature_columns, abs(model.coef_[0]).tolist())
            )

    def _validate_and_prepare_input(self, input_data: dict) -> pd.DataFrame:
        """
        Validate and prepare input data for prediction.

        Args:
            input_data: Dictionary containing input features

        Returns:
            DataFrame with validated and prepared features
        """
        required_fields = ["umur", "nilai_ipk", "nilai_skd", "nilai_skb"]
        feature_mapping = {
            "umur": "Umur",
            "nilai_ipk": "Nilai IPK",
            "nilai_skd": "Nilai SKD",
            "nilai_skb": "Nilai SKB",
        }

        # Check for missing fields
        missing = []
        missing.extend(
            field
            for field in required_fields
            if field not in input_data or input_data[field] is None
        )
        if missing:
            return {
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing)}",
                "data": None,
            }

        # Value range validation
        value_ranges = {
            "umur": (17, 60),
            "nilai_ipk": (0, 4),
            "nilai_skd": (0, 500),
            "nilai_skb": (0, 500),
        }

        try:
            validated_values = {}
            for field, (min_val, max_val) in value_ranges.items():
                try:
                    value = float(input_data[field])
                    if value < min_val or value > max_val:
                        return {
                            "status": "error",
                            "message": f"Invalid {field} value: must be between {min_val} and {max_val}",
                            "data": None,
                        }
                    validated_values[feature_mapping[field]] = value
                except (ValueError, TypeError):
                    return {
                        "status": "error",
                        "message": f"Invalid {field} value: must be a number",
                        "data": None,
                    }

            return pd.DataFrame([validated_values])[self.feature_columns]

        except Exception as e:
            return {
                "status": "error",
                "message": f"Invalid input values: {str(e)}",
                "data": None,
            }

    def _make_predictions(
        self, models: dict, scaled_features: np.ndarray, input_data: dict
    ) -> dict:
        """
        Make predictions using the ensemble of models.

        Args:
            models: Dictionary of trained models
            scaled_features: Scaled input features
            input_data: Original input data

        Returns:
            Dictionary containing prediction results and metadata
        """
        predictions = {
            "input": input_data,
            "predictions": {},
            "majority_vote": None,
            "model_performances": {},
            "metadata": {
                "model_metrics": self._meta.get("model_metrics", {}),
                "data_summary": self._meta.get("data_summary", {}),
                "class_distribution": self._meta.get("class_distribution", {}),
            },
        }

        results = []
        for name, model in models.items():
            model_key = name.lower().replace(" ", "_")
            try:
                proba = model.predict_proba(scaled_features)[0]
                pred_idx = np.argmax(proba)
                prediction = self.label_encoder.inverse_transform([pred_idx])[0]
                confidence = float(proba[pred_idx])

                predictions["predictions"][model_key] = {
                    "result": str(prediction),
                    "confidence": confidence,
                    "class_probabilities": {
                        str(self.label_encoder.classes_[i]): float(p)
                        for i, p in enumerate(proba)
                    },
                }

                # Add detailed performance metrics
                predictions["model_performances"][model_key] = {
                    "accuracy": confidence,
                    "prediction": str(prediction),
                    "metrics": self._meta.get("model_metrics", {})
                    .get("confusion_matrices", {})
                    .get(model_key, {}),
                    "roc_metrics": self._meta.get("model_metrics", {})
                    .get("roc_curves", {})
                    .get(model_key, {}),
                }

                results.append(prediction)

            except Exception as e:
                print(f"Warning: Error in model {name}: {str(e)}")
                continue

        if results:
            majority_prediction = Counter(results).most_common(1)[0][0]
            predictions["majority_vote"] = {
                "result": str(majority_prediction),
                "confidence": (results.count(majority_prediction) / len(results)),
            }

        return predictions

    def _check_models_exist(self) -> bool:
        """
        Check if the required model files exist in the specified directory.

        Returns:
            Boolean indicating whether all required model files exist
        """
        required_files = [
            os.path.join(self.model_dir, f"{name.lower().replace(' ', '_')}.joblib")
            for name in self.models.keys()
        ]
        required_files.append(os.path.join(self.model_dir, "scaler.joblib"))
        required_files.append(os.path.join(self.model_dir, "metadata.json"))
        required_files.append(os.path.join(self.model_dir, "label_encoder.joblib"))

        return all(os.path.exists(file) for file in required_files)

    def _save_models_and_meta(self) -> None:
        """
        Save trained models and metadata to the specified directory.
        """
        for name, model in self.models.items():
            joblib.dump(
                model,
                os.path.join(
                    self.model_dir, f"{name.lower().replace(' ', '_')}.joblib"
                ),
            )

        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.joblib"))
        joblib.dump(
            self.label_encoder, os.path.join(self.model_dir, "label_encoder.joblib")
        )
        self.save_meta()

    def load_models(self) -> dict:
        """
        Load trained models from the specified directory.

        Returns:
            Dictionary of loaded models
        """
        loaded_models = {}
        for name in self.models.keys():
            filename = os.path.join(
                self.model_dir, f"{name.lower().replace(' ', '_')}.joblib"
            )
            if os.path.exists(filename):
                loaded_models[name] = joblib.load(filename)
        self.models = loaded_models
        return loaded_models

    def load_scaler(self) -> bool:
        """
        Load the scaler and label encoder from the specified directory.

        Returns:
            Boolean indicating whether the scaler and label encoder were successfully loaded
        """
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        encoder_path = os.path.join(self.model_dir, "label_encoder.joblib")
        if os.path.exists(scaler_path) and os.path.exists(encoder_path):
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            return True
        return False

    def _update_meta_statistics(self, data: pd.DataFrame) -> None:
        """
        Update metadata with statistical summaries and correlations of the training data.

        Args:
            data: Cleaned training data DataFrame
        """
        # Create a copy for statistics to avoid modifying original data
        stats_data = data.copy()

        # Handle date column separately
        date_stats = None
        if "Tanggal Lahir" in stats_data.columns:
            try:
                # Create a separate series for date statistics
                date_series = pd.to_datetime(
                    stats_data["Tanggal Lahir"], errors="coerce"
                )
                timestamp_series = date_series.astype(np.int64) // 10**9
                date_stats = {
                    "min": timestamp_series.min(),
                    "max": timestamp_series.max(),
                    "mean": timestamp_series.mean(),
                    "median": timestamp_series.median(),
                }
                # Remove date column from main statistics calculation
                stats_data = stats_data.drop(columns=["Tanggal Lahir"])
            except Exception as e:
                print(f"Warning: Error processing dates: {str(e)}")
                stats_data = stats_data.drop(columns=["Tanggal Lahir"])

        # Calculate statistics for numeric columns
        self._meta.update(
            {
                "data_summary": stats_data.describe(include=["number"]).to_dict(),
                "missing_values": data.isna().sum().to_dict(),
                "class_distribution": data[self.target_column].value_counts().to_dict(),
                "correlation_matrix": data[self.feature_columns].corr().to_dict(),
            }
        )

        # Add date statistics if available
        if date_stats:
            self._meta["date_statistics"] = date_stats

    def _convert_to_serializable(self, obj):
        """
        Convert objects to JSON serializable format.

        Args:
            obj: Object to be converted

        Returns:
            JSON serializable object
        """
        if obj is None:
            return None

        # Handle different types
        if isinstance(obj, pd.DataFrame):
            return obj.replace({pd.NaT: None}).to_dict(orient="records")

        if isinstance(obj, pd.Series):
            return obj.replace({pd.NaT: None}).to_dict()

        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")

        # Updated numeric type handling for NumPy 2.0+
        if isinstance(
            obj,
            (
                np.integer,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return None if np.isnan(obj) else float(obj)

        if isinstance(obj, np.ndarray):
            return [self._convert_to_serializable(x) for x in obj]

        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d")

        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]

        return None if pd.isna(obj) else obj

    def save_meta(self) -> None:
        """
        Save metadata to the specified directory.
        """
        meta_path = os.path.join(self.model_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self._convert_to_serializable(self._meta), f, indent=2)

    def load_meta(self) -> bool:
        """
        Load metadata from the specified directory.

        Returns:
            Boolean indicating whether the metadata was successfully loaded
        """
        meta_path = os.path.join(self.model_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self._meta = json.load(f)
            return True
        return False

    def _update_prediction_history(self, predictions: dict, input_data: dict) -> None:
        """
        Update the prediction history with the latest prediction.

        Args:
            predictions: Dictionary containing prediction results
            input_data: Original input data
        """
        if "prediction_history" not in self._meta:
            self._meta["prediction_history"] = []

        prediction_record = {
            "timestamp": datetime.now().isoformat(),
            "input_data": input_data,
            "predictions": predictions["predictions"],
            "majority_vote": predictions["majority_vote"],
        }

        self._meta["prediction_history"].append(prediction_record)
        self._meta["latest_prediction"] = prediction_record
        self.save_meta()
