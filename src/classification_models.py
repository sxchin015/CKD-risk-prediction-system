"""
Classification Models Module for CKD Prediction System

This module implements and trains various classification models
for predicting Chronic Kidney Disease (CKD: Yes/No).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import joblib
import logging

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CKDClassificationModels:
    """
    Classification models for CKD prediction.
    
    Implements:
    - Logistic Regression
    - Random Forest Classifier
    - XGBoost Classifier (if available)
    - Gradient Boosting Classifier
    - MLP (Neural Network) Classifier
    """
    
    def __init__(self):
        """Initialize the classification models container."""
        self.models = {}
        self.trained_models = {}
        self.best_params = {}
        self.results = {}
        self.feature_names = None
        
        # Initialize models with default parameters
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all classification models with default parameters."""
        
        # Logistic Regression
        self.models['logistic_regression'] = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = XGBClassifier(
                n_estimators=100,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
        
        # MLP Neural Network
        self.models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        logger.info(f"Initialized {len(self.models)} classification models")
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """
        Get hyperparameter grids for each model.
        
        Returns:
            Dictionary of parameter grids for each model
        """
        grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        if XGBOOST_AVAILABLE:
            grids['xgboost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        
        return grids
    
    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Any:
        """
        Train a single model with default parameters.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            feature_names: Optional list of feature names
            
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.feature_names = feature_names
        
        logger.info(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        
        logger.info(f"{model_name} trained successfully")
        return model
    
    def train_with_hyperparameter_tuning(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
        scoring: str = 'roc_auc',
        feature_names: Optional[List[str]] = None
    ) -> Tuple[Any, Dict]:
        """
        Train a model with hyperparameter tuning using GridSearchCV.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            scoring: Scoring metric for optimization
            feature_names: Optional list of feature names
            
        Returns:
            Tuple of (best model, best parameters)
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.feature_names = feature_names
        grids = self.get_hyperparameter_grids()
        
        if model_name not in grids:
            logger.warning(f"No hyperparameter grid for {model_name}. Using default parameters.")
            return self.train_model(model_name, X_train, y_train, feature_names), {}
        
        logger.info(f"Training {model_name} with hyperparameter tuning...")
        
        grid_search = GridSearchCV(
            self.models[model_name],
            grids[model_name],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.trained_models[model_name] = grid_search.best_estimator_
        self.best_params[model_name] = grid_search.best_params_
        
        logger.info(f"{model_name} best params: {grid_search.best_params_}")
        logger.info(f"{model_name} best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparameters: bool = False,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train all classification models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform hyperparameter tuning
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of trained models
        """
        self.feature_names = feature_names
        
        for model_name in self.models.keys():
            if tune_hyperparameters:
                self.train_with_hyperparameter_tuning(
                    model_name, X_train, y_train, feature_names=feature_names
                )
            else:
                self.train_model(model_name, X_train, y_train, feature_names)
        
        return self.trained_models
    
    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate a trained model on test data.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        
        # Calculate Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
        metrics['pr_curve'] = {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
        
        self.results[model_name] = metrics
        
        return metrics
    
    def evaluate_all_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation results for all models
        """
        for model_name in self.trained_models.keys():
            self.evaluate_model(model_name, X_test, y_test)
        
        return self.results
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Create a comparison table of all model metrics.
        
        Returns:
            DataFrame with model comparison
        """
        comparison = []
        
        for model_name, metrics in self.results.items():
            comparison.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy (%)': round(metrics['accuracy'] * 100, 2),
                'Precision (%)': round(metrics['precision'] * 100, 2),
                'Recall (%)': round(metrics['recall'] * 100, 2),
                'F1-Score (%)': round(metrics['f1'] * 100, 2),
                'ROC-AUC (%)': round(metrics['roc_auc'] * 100, 2)
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('ROC-AUC (%)', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, Any]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model name, model object)
        """
        best_model = None
        best_score = -1
        best_name = None
        
        for model_name, metrics in self.results.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_name = model_name
                best_model = self.trained_models[model_name]
        
        logger.info(f"Best model: {best_name} with {metric}: {best_score:.4f}")
        
        return best_name, best_model
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model to use
            X: Features for prediction
            
        Returns:
            Predicted class labels
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        return self.trained_models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities using a trained model.
        
        Args:
            model_name: Name of the model to use
            X: Features for prediction
            
        Returns:
            Prediction probabilities
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        return self.trained_models[model_name].predict_proba(X)
    
    def get_feature_importance(self, model_name: str) -> Optional[pd.DataFrame]:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with feature importances or None if not applicable
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def save_models(self, path: str) -> None:
        """
        Save all trained models to disk.
        
        Args:
            path: Directory path to save models
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_file = save_path / f'classification_{model_name}.pkl'
            joblib.dump(model, model_file)
            logger.info(f"Saved {model_name} to {model_file}")
        
        # Save best params
        joblib.dump(self.best_params, save_path / 'classification_best_params.pkl')
    
    def load_models(self, path: str) -> None:
        """
        Load trained models from disk.
        
        Args:
            path: Directory path containing saved models
        """
        load_path = Path(path)
        
        for model_name in self.models.keys():
            model_file = load_path / f'classification_{model_name}.pkl'
            if model_file.exists():
                self.trained_models[model_name] = joblib.load(model_file)
                logger.info(f"Loaded {model_name} from {model_file}")
        
        # Load best params
        params_file = load_path / 'classification_best_params.pkl'
        if params_file.exists():
            self.best_params = joblib.load(params_file)


def run_classification_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    tune_hyperparameters: bool = False,
    save_path: Optional[str] = None
) -> Tuple[CKDClassificationModels, pd.DataFrame]:
    """
    Run the complete classification pipeline.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        feature_names: Optional list of feature names
        tune_hyperparameters: Whether to perform hyperparameter tuning
        save_path: Optional path to save models
        
    Returns:
        Tuple of (models object, comparison DataFrame)
    """
    # Initialize models
    classifier = CKDClassificationModels()
    
    # Train all models
    classifier.train_all_models(
        X_train, y_train,
        tune_hyperparameters=tune_hyperparameters,
        feature_names=feature_names
    )
    
    # Evaluate all models
    classifier.evaluate_all_models(X_test, y_test)
    
    # Get comparison table
    comparison = classifier.get_comparison_table()
    
    # Print results
    print("\n" + "="*60)
    print("CLASSIFICATION MODEL COMPARISON")
    print("="*60)
    print(comparison.to_string(index=False))
    
    # Get best model
    best_name, _ = classifier.get_best_model()
    print(f"\nBest Model: {best_name.replace('_', ' ').title()}")
    
    # Save models if path provided
    if save_path:
        classifier.save_models(save_path)
    
    return classifier, comparison


if __name__ == "__main__":
    from data_pipeline import CKDDataPipeline
    
    # Load and prepare data
    pipeline = CKDDataPipeline()
    X_train, X_test, y_train, y_test, feature_names = pipeline.get_classification_data()
    
    # Run classification pipeline
    classifier, comparison = run_classification_pipeline(
        X_train, X_test, y_train, y_test,
        feature_names=feature_names,
        tune_hyperparameters=False
    )
