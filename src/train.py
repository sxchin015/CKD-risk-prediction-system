"""
Main Training Script for CKD Prediction System

Uses kidney_disease.csv:
- 80/20 train/test split
- Trains: RandomForest, XGBoost (if available), LogisticRegression
- Evaluates: Accuracy, Precision, Recall, F1, ROC-AUC
- Selects best classifier by ROC-AUC
- Saves: best_classifier.pkl, model_info.pkl
- Optional: trains RandomForestRegressor if regression target exists
"""

import sys
import logging
from pathlib import Path
import warnings
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline import CKDDataPipeline
from classification_models import CKDClassificationModels
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Only these classifiers (user requirement)
CLASSIFICATION_MODELS = ["random_forest", "logistic_regression"]
if True:  # XGBoost check done in classification_models
    try:
        from xgboost import XGBClassifier
        CLASSIFICATION_MODELS.insert(1, "xgboost")
    except ImportError:
        pass


def train_complete_pipeline(
    data_path: str = None,
    models_path: str = None,
    tune_hyperparameters: bool = False,
    save_figures: bool = True,
    target_samples: int = 10000,
):
    """Run the complete training pipeline."""

    project_root = Path(__file__).parent.parent
    if data_path is None:
        for p in ["kidney_disease.csv", "data/kidney_disease.csv"]:
            cand = project_root / p
            if cand.exists():
                data_path = str(cand)
                break
        if data_path is None:
            raise FileNotFoundError("kidney_disease.csv not found in project root or data/")

    if models_path is None:
        models_path = project_root / "models"
    models_path = Path(models_path)
    figures_path = models_path / "figures"
    models_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CKD PREDICTION SYSTEM - TRAINING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load and prepare data
    logger.info("\n[STEP 1] Loading and preparing data...")
    pipeline = CKDDataPipeline(str(data_path))
    pipeline.load_data()

    X_train, X_test, y_train, y_test, feature_names = pipeline.get_classification_data(
        test_size=0.2,
        random_state=42,
        target_samples=target_samples,
    )
    logger.info(f"Classification data - Train: {X_train.shape}, Test: {X_test.shape}")

    # Save pipeline first
    pipeline.save_pipeline(str(models_path))
    logger.info("Data pipeline saved.")

    # Step 2: Train classification models (RF, XGB, LogisticRegression only)
    logger.info("\n[STEP 2] Training classification models...")
    classifier = CKDClassificationModels()

    for model_name in CLASSIFICATION_MODELS:
        if model_name not in classifier.models:
            logger.warning(f"Skipping {model_name} (not available)")
            continue
        if tune_hyperparameters:
            classifier.train_with_hyperparameter_tuning(
                model_name, X_train, y_train,
                feature_names=feature_names,
                scoring="roc_auc",
            )
        else:
            classifier.train_model(
                model_name, X_train, y_train,
                feature_names=feature_names,
            )

    classifier.evaluate_all_models(X_test, y_test)
    cls_comparison = classifier.get_comparison_table()
    best_cls_name, best_cls_model = classifier.get_best_model("roc_auc")
    logger.info(f"Best classification model: {best_cls_name}")

    # Step 3: Regression (only if regression target exists)
    reg_comparison = None
    best_reg_name = None
    best_reg_model = None
    X_test_reg = np.array([])
    y_test_reg = np.array([])
    has_regression = pipeline.has_regression_target()

    if has_regression:
        logger.info("\n[STEP 3] Training regression model (RandomForestRegressor)...")
        try:
            X_train_reg, X_test_reg, y_train_reg, y_test_reg, _ = pipeline.get_regression_data(
                test_size=0.2, random_state=42
            )
            regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            regressor.fit(X_train_reg, y_train_reg)
            y_pred_reg = regressor.predict(X_test_reg)
            rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
            r2 = r2_score(y_test_reg, y_pred_reg)
            reg_comparison = pd.DataFrame([
                {"Model": "Random Forest", "RMSE": rmse, "R²": r2}
            ])
            best_reg_name = "random_forest"
            best_reg_model = regressor
            joblib.dump(best_reg_model, str(models_path / "best_regressor.pkl"))
            logger.info(f"Regression - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        except Exception as e:
            logger.warning(f"Regression training failed: {e}")
            has_regression = False
            X_test_reg = np.array([])
            y_test_reg = np.array([])
    else:
        logger.info("\n[STEP 3] No regression target in dataset - skipping regression.")

    # Step 4: Evaluation figures (optional)
    if save_figures:
        try:
            from model_evaluation import ModelEvaluator
            evaluator = ModelEvaluator()
            reg_results = {}
            if has_regression and best_reg_model is not None:
                y_pred_reg = best_reg_model.predict(X_test_reg)
                reg_results = {
                    "random_forest": {
                        "y_pred": y_pred_reg,
                        "y_test": y_test_reg,
                        "rmse": np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)),
                        "r2": r2_score(y_test_reg, y_pred_reg),
                    }
                }
            evaluator.generate_evaluation_report(
                classifier.results,
                reg_results,
                y_test,
                y_test_reg,
                cls_comparison,
                reg_comparison if reg_comparison is not None else pd.DataFrame(),
                save_path=str(figures_path),
            )
        except Exception as e:
            logger.warning(f"Could not generate figures: {e}")

    # Step 5: Save best models and model_info
    logger.info("\n[STEP 5] Saving models...")
    joblib.dump(best_cls_model, str(models_path / "best_classifier.pkl"))

    model_info = {
        "best_classifier": best_cls_name,
        "best_regressor": best_reg_name,
        "feature_names": feature_names,
        "classification_metrics": cls_comparison.to_dict() if cls_comparison is not None else {},
        "regression_metrics": reg_comparison.to_dict() if reg_comparison is not None else {},
        "has_regression": has_regression,
    }
    joblib.dump(model_info, str(models_path / "model_info.pkl"))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(cls_comparison.to_string(index=False))
    print(f"\nBest Model: {best_cls_name.replace('_', ' ').title()}")
    if has_regression and reg_comparison is not None:
        print("\n" + "=" * 60)
        print("REGRESSION RESULTS")
        print("=" * 60)
        print(reg_comparison.to_string(index=False))
    print(f"\nModels saved to: {models_path}")

    return {
        "classifier": classifier,
        "pipeline": pipeline,
        "best_classifier": (best_cls_name, best_cls_model),
        "best_regressor": (best_reg_name, best_reg_model) if has_regression else (None, None),
        "classification_comparison": cls_comparison,
        "regression_comparison": reg_comparison,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CKD Prediction Models")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--no-figures", action="store_true", help="Skip saving figures")
    parser.add_argument("--samples", type=int, default=10000, help="Target number of samples (oversample if needed)")
    args = parser.parse_args()

    train_complete_pipeline(
        tune_hyperparameters=args.tune,
        save_figures=not args.no_figures,
        target_samples=args.samples,
    )
