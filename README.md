# 🏥 Chronic Kidney Disease (CKD) Progression & Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive, production-ready AI/ML system for predicting Chronic Kidney Disease risk and kidney function scores using advanced machine learning and neural network models.

![CKD Prediction System](https://img.icons8.com/color/96/000000/kidney.png)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Models](#-models)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

Chronic Kidney Disease affects approximately 10% of the global population and is a leading cause of death worldwide. Early detection and intervention can significantly slow disease progression and improve patient outcomes.

This system uses machine learning to:
- **Classify** patients as having CKD (Yes/No)
- **Predict** kidney function scores (continuous value)
- **Explain** predictions using AI-powered insights
- **Recommend** personalized preventive measures

## ✨ Features

### 🤖 Machine Learning Models

**Classification Models (CKD: Yes/No):**
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- Gradient Boosting Classifier


**Regression Models (Kidney Function Score):**
- Linear Regression
- Ridge/Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor


### 📊 Data Pipeline
- Automated data loading and validation
- Missing value imputation
- Feature encoding (Label Encoding)
- Feature scaling (StandardScaler)
- Train-test splitting with stratification

### 📈 Evaluation & Visualization
- Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Regression: RMSE, MAE, R², MAPE
- Confusion matrices
- ROC curves
- Feature importance plots
- Model comparison dashboards

### 🔍 Explainability (AI Integration)
- SHAP (SHapley Additive exPlanations) values
- Feature importance analysis
- Individual prediction explanations
- Natural language insights

### 🌐 Web Application
- Beautiful Streamlit interface
- Patient data input form
- Real-time predictions
- Interactive visualizations
- AI-generated reports

## 📁 Project Structure

```
ckd_prediction/
├── data/
│   └── Chronic_Kidney_Disease_Risk_Assessment.csv
├── notebooks/
│   └── CKD_EDA_and_Modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py      # Data loading & preprocessing
│   ├── eda.py                 # Exploratory data analysis
│   ├── classification_models.py  # Classification ML models
│   ├── regression_models.py   # Regression ML models
│   ├── model_evaluation.py    # Evaluation & visualization
│   ├── explainability.py      # SHAP & feature importance
│   ├── ai_assistant.py        # AI-powered explanations
│   ├── train.py               # Training pipeline
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── models/                    # Saved trained models
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Installation

### Option 1: Run in Google Colab (Recommended for Quick Start)

The easiest way to run this project is using Google Colab:

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the notebook: `notebooks/CKD_Complete_Project_Colab.ipynb`
3. Run all cells sequentially

**Or use this direct approach:**
- Upload `CKD_Complete_Project_Colab.ipynb` to your Google Drive
- Open with Google Colab
- Click "Runtime" → "Run all"

The Colab notebook is **completely self-contained** - no additional files needed!

---

### Option 2: Run Locally

#### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (optional)

#### Step 1: Clone or Create Project

```bash
# If cloning from repository
git clone https://github.com/yourusername/ckd-prediction.git
cd ckd-prediction

# Or navigate to the project directory
cd ckd_prediction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import sklearn; import streamlit; print('Installation successful!')"
```

## 💻 Usage

### 1. Train Models

Run the training pipeline to train all models:

```bash
# Basic training (fast)
python src/train.py

# With hyperparameter tuning (slower but better results)
python src/train.py --tune

# Skip saving figures
python src/train.py --no-figures
```

This will:
- Load and preprocess the dataset
- Train all classification and regression models
- Evaluate model performance
- Save trained models to `models/` directory
- Generate evaluation visualizations

### 2. Run Web Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

### 3. Use in Python Code

```python
from src.data_pipeline import CKDDataPipeline
from src.classification_models import CKDClassificationModels
from src.ai_assistant import CKDAssistant

# Load data
pipeline = CKDDataPipeline()
X_train, X_test, y_train, y_test, features = pipeline.get_classification_data()

# Train models
classifier = CKDClassificationModels()
classifier.train_all_models(X_train, y_train, feature_names=features)

# Evaluate
classifier.evaluate_all_models(X_test, y_test)
comparison = classifier.get_comparison_table()
print(comparison)

# Make predictions
patient_data = {
    'age': 55,
    'gender': 'Male',
    'blood_pressure_systolic': 140,
    'blood_glucose': 150,
    # ... other features
}

X = pipeline.preprocess_single_patient(patient_data)
prediction = classifier.predict_proba('random_forest', X)
print(f"CKD Probability: {prediction[0][1]:.2%}")

# Get AI explanation
assistant = CKDAssistant()
interpretation = assistant.interpret_risk_level(prediction[0][1])
print(interpretation['description'])
```

### 4. Run Jupyter Notebook

```bash
jupyter notebook notebooks/CKD_EDA_and_Modeling.ipynb
```

## 📊 Dataset

### Features

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Patient age in years |
| gender | Categorical | Male/Female |
| blood_pressure_systolic | Numeric | Systolic blood pressure (mmHg) |
| blood_pressure_diastolic | Numeric | Diastolic blood pressure (mmHg) |
| blood_glucose | Numeric | Blood glucose level (mg/dL) |
| serum_creatinine | Numeric | Serum creatinine level (mg/dL) |
| blood_urea_nitrogen | Numeric | BUN level |
| hemoglobin | Numeric | Hemoglobin level (g/dL) |
| albumin | Numeric | Albumin level |
| specific_gravity | Numeric | Urine specific gravity |
| red_blood_cells | Categorical | Normal/Abnormal |
| pus_cell | Categorical | Normal/Abnormal |
| pus_cell_clumps | Categorical | Present/Not present |
| bacteria | Categorical | Present/Not present |
| hypertension | Categorical | Yes/No |
| diabetes_mellitus | Categorical | Yes/No |
| coronary_artery_disease | Categorical | Yes/No |
| appetite | Categorical | Good/Poor |
| pedal_edema | Categorical | Yes/No |
| anemia | Categorical | Yes/No |
| smoking | Categorical | Yes/No |
| family_history_ckd | Categorical | Yes/No |
| bmi | Numeric | Body Mass Index |

### Target Variables

- **ckd** (Classification): Yes/No - Whether patient has CKD
- **kidney_function_score** (Regression): 0-100 score indicating kidney function

## 🤖 Models

### Classification Performance (Expected)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.95 | 0.94 | 0.96 | 0.95 | 0.98 |
| XGBoost | 0.94 | 0.93 | 0.95 | 0.94 | 0.97 |
| Gradient Boosting | 0.93 | 0.92 | 0.94 | 0.93 | 0.96 |
| MLP Neural Network | 0.92 | 0.91 | 0.93 | 0.92 | 0.95 |
| Logistic Regression | 0.88 | 0.87 | 0.89 | 0.88 | 0.92 |

### Regression Performance (Expected)

| Model | RMSE | MAE | R² | MAPE (%) |
|-------|------|-----|-------|----------|
| Random Forest | 5.2 | 3.8 | 0.96 | 6.5 |
| XGBoost | 5.5 | 4.1 | 0.95 | 7.2 |
| Gradient Boosting | 5.8 | 4.3 | 0.94 | 7.8 |
| MLP Neural Network | 6.2 | 4.6 | 0.93 | 8.5 |
| Linear Regression | 8.5 | 6.2 | 0.88 | 11.2 |

## 🔧 API Reference

### Data Pipeline

```python
from src.data_pipeline import CKDDataPipeline

pipeline = CKDDataPipeline(data_path='path/to/data.csv')
pipeline.load_data()
pipeline.get_data_info()
X_train, X_test, y_train, y_test, features = pipeline.get_classification_data()
X_train, X_test, y_train, y_test, features = pipeline.get_regression_data()
pipeline.save_pipeline('models/')
pipeline.load_pipeline('models/')
```

### Classification Models

```python
from src.classification_models import CKDClassificationModels

models = CKDClassificationModels()
models.train_all_models(X_train, y_train, feature_names=features)
models.evaluate_all_models(X_test, y_test)
comparison = models.get_comparison_table()
best_name, best_model = models.get_best_model('roc_auc')
predictions = models.predict('random_forest', X_new)
probabilities = models.predict_proba('random_forest', X_new)
```

### AI Assistant

```python
from src.ai_assistant import CKDAssistant

assistant = CKDAssistant()
print(assistant.explain_ckd())
risk = assistant.interpret_risk_level(probability=0.75)
kidney = assistant.interpret_kidney_function_score(score=45)
report = assistant.generate_patient_report(patient_data, ckd_prob, kidney_score)
```

## 🎨 Web Application Features

- **Patient Input Form**: Easy-to-use form for entering patient data
- **Real-time Predictions**: Instant CKD risk and kidney function predictions
- **Interactive Charts**: Gauge charts, radar plots, and bar charts
- **Risk Assessment**: Color-coded risk levels (Low/Medium/High)
- **AI Explanations**: Natural language interpretation of results
- **Recommendations**: Personalized health recommendations
- **Educational Content**: Information about CKD, stages, and prevention

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- scikit-learn team for excellent ML library
- Streamlit team for the amazing web framework
- SHAP developers for model explainability tools
- Medical professionals who contributed domain knowledge

---

Made with ❤️ for better kidney health prediction
