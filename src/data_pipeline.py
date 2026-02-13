"""
Data Pipeline Module for CKD Prediction System

Handles data loading, cleaning, preprocessing for kidney_disease.csv:
- Automatic column type detection (numerical, categorical, target)
- Missing value handling (median for numerical, mode for categorical)
- OneHotEncoder for categorical features
- StandardScaler for numerical features
- ColumnTransformer for unified preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging

# Optional: SMOTE for oversampling to reach target sample size
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logging.warning("imblearn not available. Install with: pip install imbalanced-learn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default data paths to check (project root, then data/)
DEFAULT_DATA_PATHS = [
    "kidney_disease.csv",
    "data/kidney_disease.csv",
]

# Target column candidates (first match wins)
TARGET_CANDIDATES = ["ckd", "classification", "class"]

# Regression target candidates (eGFR, kidney score, etc.)
REGRESSION_TARGET_CANDIDATES = ["egfr", "kidney_function_score", "gfr", "ckd_stage"]

# User-friendly column rename mapping (original -> display name)
COLUMN_RENAME_MAP = {
    "age": "age",
    "bp": "blood_pressure",
    "sg": "specific_gravity",
    "al": "albumin",
    "su": "sugar",
    "rbc": "red_blood_cells",
    "pc": "pus_cell",
    "pcc": "pus_cell_clumps",
    "ba": "bacteria",
    "bgr": "blood_glucose_random",
    "bu": "blood_urea",
    "sc": "serum_creatinine",
    "sod": "sodium",
    "pot": "potassium",
    "hemo": "haemoglobin",
    "pcv": "packed_cell_volume",
    "wc": "white_blood_cell_count",
    "rc": "red_blood_cell_count",
    "htn": "hypertension",
    "dm": "diabetes_mellitus",
    "cad": "coronary_artery_disease",
    "appet": "appetite",
    "pe": "peda_edema",
    "ane": "aanemia",
    "classification": "class",
}


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names: strip whitespace, lowercase, replace spaces with underscores."""
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]
    return df


def _clean_string_values(series: pd.Series) -> pd.Series:
    """Strip whitespace from string/categorical values."""
    if series.dtype == object or series.dtype.name == "category":
        return series.astype(str).str.strip().str.lower()
    return series


def _detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect classification target column (ckd, classification, etc.)."""
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _detect_regression_target(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect regression target column if present."""
    for candidate in REGRESSION_TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _detect_column_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """
    Detect numerical vs categorical columns.
    Excludes target, id, and non-feature columns.
    """
    exclude = {target_col, "id"}
    feature_cols = [c for c in df.columns if c not in exclude]

    numerical = []
    categorical = []

    for col in feature_cols:
        if df[col].dtype in (np.float64, np.int64, np.float32, np.int32):
            numerical.append(col)
        elif df[col].dtype == object or df[col].dtype.name == "category":
            # Could be categorical or numeric stored as string
            try:
                pd.to_numeric(df[col], errors="raise")
                numerical.append(col)
            except (ValueError, TypeError):
                categorical.append(col)
        else:
            categorical.append(col)

    return numerical, categorical


class CKDDataPipeline:
    """
    Data pipeline for kidney_disease.csv.
    Auto-detects columns, handles missing values, encodes and scales features.
    """

    def __init__(self, data_path: Optional[str] = None, project_root: Optional[Path] = None):
        self.data_path = data_path
        self.project_root = project_root or Path(__file__).parent.parent
        self.raw_data: Optional[pd.DataFrame] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.target_column: Optional[str] = None
        self.regression_target_column: Optional[str] = None
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_names_out: List[str] = []
        self.label_encoder_target: Optional[Any] = None  # For encoding ckd/notckd -> 0/1

    def _resolve_data_path(self, path: Optional[str] = None) -> Path:
        """Resolve path to kidney_disease.csv (project root or data/)."""
        if path:
            p = Path(path)
            if p.exists():
                return p
        for rel in DEFAULT_DATA_PATHS:
            full = self.project_root / rel
            if full.exists():
                return full
        raise FileNotFoundError(
            f"kidney_disease.csv not found. Tried: {[str(self.project_root / x) for x in DEFAULT_DATA_PATHS]}"
        )

    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load kidney_disease.csv from project root or data/."""
        data_path = self._resolve_data_path(path or self.data_path)
        logger.info(f"Loading data from: {data_path}")

        self.raw_data = pd.read_csv(data_path)
        self.raw_data = _clean_column_names(self.raw_data)
        # Apply user-friendly column names
        rename = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in self.raw_data.columns}
        self.raw_data = self.raw_data.rename(columns=rename)
        logger.info(f"Data loaded. Shape: {self.raw_data.shape}, Columns: {list(self.raw_data.columns)}")
        return self.raw_data

    def _prepare_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean string values and convert numeric columns."""
        data = df.copy()

        # Drop id if present
        if "id" in data.columns:
            data = data.drop(columns=["id"])

        # Clean categorical/object columns
        for col in data.columns:
            if data[col].dtype == object:
                data[col] = _clean_string_values(data[col])
                # Replace empty string with NaN
                data[col] = data[col].replace("", np.nan)
                data[col] = data[col].replace("nan", np.nan)
                data[col] = data[col].replace("?", np.nan)

        # Convert numeric columns that might be stored as object
        for col in data.columns:
            if col in (self.target_column or "", self.regression_target_column or ""):
                continue
            if data[col].dtype == object:
                try:
                    data[col] = pd.to_numeric(data[col])
                except (ValueError, TypeError):
                    pass  # Leave as object (categorical)

        return data

    def fit(self, df: Optional[pd.DataFrame] = None, target_col: Optional[str] = None) -> "CKDDataPipeline":
        """
        Fit the preprocessing pipeline (imputation, encoding, scaling).
        """
        data = df if df is not None else self.raw_data
        if data is None:
            raise ValueError("No data. Call load_data() first.")

        data = self._prepare_raw_data(data)

        # Detect target
        self.target_column = target_col or _detect_target_column(data)
        if not self.target_column:
            raise ValueError("Could not detect target column. Expected one of: " + str(TARGET_CANDIDATES))
        self.regression_target_column = _detect_regression_target(data)

        # Detect column types
        self.numerical_features, self.categorical_features = _detect_column_types(
            data, self.target_column
        )
        # Exclude regression target from features if present
        if self.regression_target_column and self.regression_target_column in self.numerical_features:
            self.numerical_features = [c for c in self.numerical_features if c != self.regression_target_column]

        logger.info(f"Target: {self.target_column}")
        logger.info(f"Numerical: {self.numerical_features}")
        logger.info(f"Categorical: {self.categorical_features}")

        # Store categorical unique values for app input options
        self._categorical_options = {}
        for col in self.categorical_features:
            if col in data.columns:
                opts = sorted(data[col].dropna().astype(str).str.strip().str.lower().unique().tolist())
                self._categorical_options[col] = [x for x in opts if x and x != "nan"]

        # Encode target: ckd=1 (positive), notckd/other=0
        y_raw = data[self.target_column].astype(str).str.strip().str.lower()
        self.label_encoder_target = None  # We use explicit mapping
        self._target_classes = sorted(y_raw.dropna().unique())

        # Build ColumnTransformer
        transformers = []

        if self.numerical_features:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipeline, self.numerical_features))

        if self.categorical_features:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipeline, self.categorical_features))

        self.preprocessor = ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

        X = data[[c for c in self.numerical_features + self.categorical_features if c in data.columns]]
        self.preprocessor.fit(X)

        # Get output feature names
        self.feature_names_out = self._get_feature_names_out()
        logger.info(f"Preprocessor fitted. Output features: {len(self.feature_names_out)}")
        return self

    def _get_feature_names_out(self) -> List[str]:
        """Get names of transformed output features."""
        names = []
        if "num" in self.preprocessor.named_transformers_:
            names.extend(self.numerical_features)
        if "cat" in self.preprocessor.named_transformers_:
            cat_trans = self.preprocessor.named_transformers_["cat"]
            ohe = cat_trans.named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(self.categorical_features)
            names.extend(list(cat_names))
        return names

    def transform(self, df: pd.DataFrame, include_target: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform data through the fitted pipeline."""
        data = self._prepare_raw_data(df)
        X = data[[c for c in self.numerical_features + self.categorical_features if c in data.columns]]
        X_t = self.preprocessor.transform(X)

        y = None
        if include_target and self.target_column in data.columns:
            y_raw = data[self.target_column].astype(str).str.strip().str.lower()

            def _to_binary(v):
                v = str(v).strip().lower()
                # ckd = 1 (positive), notckd or other = 0
                return 1 if v == "ckd" or (v != "notckd" and "ckd" in v and "not" not in v) else 0

            y = np.array([_to_binary(v) for v in y_raw])
        return X_t, y

    def get_classification_data(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        target_samples: Optional[int] = 10000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare classification data: fit (if needed), split 80/20, optionally oversample.
        """
        if self.raw_data is None:
            self.load_data()
        if self.preprocessor is None:
            self.fit()

        X, y = self.transform(self.raw_data, include_target=True)
        if y is None:
            raise ValueError("Target column not found or empty.")

        # Oversample to target_samples if requested
        if target_samples and len(X) < target_samples:
            if SMOTE_AVAILABLE:
                try:
                    counts = np.bincount(y)
                    k = min(5, counts.min() - 1) if counts.min() > 1 else 1
                    smote = SMOTE(sampling_strategy="auto", random_state=random_state, k_neighbors=max(1, k))
                    X, y = smote.fit_resample(X, y)
                except Exception as e:
                    logger.warning(f"SMOTE failed: {e}. Using random oversampling.")
            # Random oversampling to reach target_samples
            rng = np.random.RandomState(random_state)
            while len(X) < target_samples:
                idx = rng.choice(len(X), size=min(target_samples - len(X), len(X)))
                X = np.vstack([X, X[idx]])
                y = np.concatenate([y, y[idx]])
            logger.info(f"Oversampled to {len(X)} samples")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Classification data: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, X_test, y_train, y_test, self.feature_names_out

    def get_regression_data(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare regression data if regression target exists."""
        if not self.regression_target_column:
            raise ValueError("No regression target column found in dataset.")
        if self.raw_data is None:
            self.load_data()
        if self.preprocessor is None:
            self.fit()

        X, _ = self.transform(self.raw_data, include_target=False)
        y = self.raw_data[self.regression_target_column].astype(float)
        valid = ~y.isna()
        X, y = X[valid], y.values[valid]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, self.feature_names_out

    def has_regression_target(self) -> bool:
        """Check if dataset has a regression target."""
        return self.regression_target_column is not None

    def preprocess_single_patient(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess a single patient dict for prediction. Keys must match dataset columns."""
        # Use cleaned column names
        clean_keys = {k.strip().lower().replace(" ", "_"): k for k in patient_data.keys()}
        clean_data = {}
        for clean_name, orig_key in clean_keys.items():
            clean_data[clean_name] = patient_data[orig_key]

        df = pd.DataFrame([clean_data])
        # Ensure columns align; missing cols get NaN (imputer will fill)
        for c in self.numerical_features + self.categorical_features:
            if c not in df.columns:
                df[c] = np.nan
        df = df[[c for c in self.numerical_features + self.categorical_features if c in df.columns]]
        X_t, _ = self.transform(df, include_target=False)
        return X_t

    def get_feature_schema(self) -> Dict[str, Any]:
        """Return schema for UI: numerical and categorical feature config."""
        cat_options = getattr(self, "_categorical_options", {})
        return {
            "numerical": self.numerical_features,
            "categorical": self.categorical_features,
            "categorical_options": cat_options,
            "target": self.target_column,
            "regression_target": self.regression_target_column,
            "feature_names": self.feature_names_out,
        }

    def save_pipeline(self, path: str) -> None:
        """Save fitted pipeline and metadata."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, save_path / "preprocessor.pkl")
        metadata = {
            "target_column": self.target_column,
            "regression_target_column": self.regression_target_column,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "feature_names_out": self.feature_names_out,
            "target_classes": getattr(self, "_target_classes", []),
            "categorical_options": getattr(self, "_categorical_options", {}),
        }
        joblib.dump(metadata, save_path / "pipeline_metadata.pkl")
        logger.info(f"Pipeline saved to {save_path}")

    def load_pipeline(self, path: str) -> None:
        """Load saved pipeline and metadata."""
        load_path = Path(path)
        self.preprocessor = joblib.load(load_path / "preprocessor.pkl")
        metadata = joblib.load(load_path / "pipeline_metadata.pkl")
        self.target_column = metadata["target_column"]
        self.regression_target_column = metadata.get("regression_target_column")
        self.numerical_features = metadata["numerical_features"]
        self.categorical_features = metadata["categorical_features"]
        self.feature_names_out = metadata["feature_names_out"]
        self.label_encoder_target = None
        self._target_classes = metadata.get("target_classes", [])
        self._categorical_options = metadata.get("categorical_options", {})
        logger.info(f"Pipeline loaded from {load_path}")


if __name__ == "__main__":
    pipeline = CKDDataPipeline()
    pipeline.load_data()
    X_train, X_test, y_train, y_test, features = pipeline.get_classification_data(target_samples=10000)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, Features: {len(features)}")
    schema = pipeline.get_feature_schema()
    print("Schema:", schema)
