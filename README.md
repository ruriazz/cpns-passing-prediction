# CPNS Passing Prediction API

A machine learning-based API service for predicting CPNS (Indonesian Civil Servant) recruitment outcomes using ensemble learning techniques.

## Features

- Multiple ML models integration (SVM, Decision Tree, Random Forest, k-NN, Naïve Bayes)
- Ensemble prediction with majority voting
- Model performance tracking and metrics
- Automatic data preprocessing and scaling
- Handling class imbalance using SMOTE
- RESTful API endpoints for prediction and model training
- Comprehensive model metadata and statistics

## Technical Stack

- Python 3.x
- Flask (Web Framework)
- Scikit-learn (Machine Learning)
- Pandas & NumPy (Data Processing)
- Joblib (Model Persistence)
- SMOTE (Imbalanced Learning)

## Prerequisites

- Python 3.x
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ruriazz/cpns-passing-prediction.git
cd cpns-passing-prediction
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p src/models/.trained .tmp
```

## Configuration

Create a `.env` file in the root directory:

```env
FLASK_APP=wsgi.py
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_HOST=0.0.0.0
FLASK_PORT=8000
```

## Running the Application

1. Start the Flask server:
```bash
python wsgi.py
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Prediction Endpoint
- **URL**: `/api/predict`
- **Method**: `POST`
- **Request Body**:
```json
{
    "no_peserta": "123456",     // optional
    "nama": "John Doe",         // optional
    "umur": 25,
    "nilai_ipk": 3.5,
    "nilai_skd": 350,
    "nilai_skb": 400
}
```

### 2. Training Endpoint
- **URL**: `/api/train`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: Excel file (.xlsx) containing training data

### API Response Structure

The prediction endpoint returns a detailed JSON response with the following structure:

#### Top Level Properties

- `status`: Request status ("success" or "error")
- `message`: Response message describing the outcome
- `data`: Contains all prediction results and metadata

#### Data Object Properties

1. **confusion_matrices**: Contains confusion matrix data for each model, showing:
   - `false_negative`: Number of false negative predictions
   - `false_positive`: Number of false positive predictions
   - `true_negative`: Number of true negative predictions
   - `true_positive`: Number of true positive predictions

2. **input**: Echo of the input data provided in the request

3. **majority_vote**: Final prediction result based on ensemble voting
   - Possible values: "P/L" (Pass), "TH" (Hold), "TL" (Not Pass), "TMS-1" (Not Eligible)

4. **metadata**: Contains model statistics and performance metrics:
   - `class_distribution`: Distribution of outcomes in training data
   - `correlation_matrix`: Feature correlation statistics
   - `data_summary`: Statistical summary of training data
   - `feature_importance`: Importance weights of each feature per model
   - `model_metrics`: Detailed model performance metrics

5. **model_comparison**:
   - `average_accuracy`: Mean accuracy across all models
   - `best_model`: Identifier of the best performing model

6. **model_performances**: Per-model performance metrics:
   - `accuracy`: Model accuracy score
   - `f1_score`: F1 score
   - `precision`: Precision score
   - `recall`: Recall score

7. **predictions**: Individual model predictions:
   - `class_probabilities`: Probability scores for each possible outcome
   - `confidence`: Confidence score of the prediction
   - `result`: Predicted outcome

Example Response:
```json
{
    "status": "success",
    "message": "Prediction completed successfully",
    "data": {
        "majority_vote": "TL",
        "predictions": {
            "decision_tree": {
                "result": "TL",
                "confidence": 1.0,
                "class_probabilities": {
                    "P/L": 0.0,
                    "TH": 0.0,
                    "TL": 1.0,
                    "TMS-1": 0.0
                }
            }
            // ... other models
        }
        // ... other data
    }
}
```

## Model Training Data Format

The training data should be an Excel file with the following columns:
- Umur (Age)
- Nilai IPK (GPA Score)
- Nilai SKD (Basic Competency Test Score)
- Nilai SKB (Field Competency Test Score)
- Keterangan (Status/Label)

## Project Structure

```
cpns-passing-prediction/
├── src/
│   ├── api/
│   │   ├── validators/
│   │   └── routes.py
│   ├── core/
│   │   └── config.py
│   ├── ml/
|   |   └── models/
│   │       └── prediction.py
│   ├── services/
│   │   └── prediction_service.py
│   └── app.py
├── .tmp/
├── requirements.txt
├── wsgi.py
└── README.md
```

## Features in Detail

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Outlier handling using IQR method
   - Feature scaling using StandardScaler
   - Class imbalance handling with SMOTE

2. **Model Ensemble**
   - Linear SVM
   - Decision Tree
   - Random Forest
   - k-Nearest Neighbors
   - Naïve Bayes

3. **Performance Metrics**
   - Confusion matrices
   - ROC curves
   - Feature importance
   - Cross-validation scores

## Error Handling

The API includes comprehensive error handling for:
- Invalid input data
- Missing required fields
- File format validation
- Model training errors
- Prediction errors

## Development

For development purposes, you can modify the configuration in `src/core/config.py` and add new features through the modular architecture.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](License) file for details.

## Third-Party Dependencies

This project uses various third-party libraries that are listed in the [NOTICE](NOTICE) file. Key dependencies include:
- Flask 3.1.0 (BSD License)
- Scikit-learn 1.6.1 (BSD License)
- Pandas 2.2.3 (BSD License)
- NumPy 2.2.2 (BSD License)
- imbalanced-learn 0.13.0 (MIT License)

For a complete list of dependencies and their licenses, please refer to the [NOTICE](NOTICE) file.

MIT License

Copyright (c) 2025 ruriazz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
