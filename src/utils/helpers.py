"""
Helper utility functions for the CKD Prediction System.
"""

import os
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory path."""
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """Get the models directory path."""
    return get_project_root() / "models"


def ensure_dir_exists(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """Get current timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_metric(name: str, value: float, decimals: int = 4) -> str:
    """Format a metric for display."""
    return f"{name}: {value:.{decimals}f}"


def get_risk_category(probability: float) -> str:
    """
    Categorize CKD risk based on prediction probability.
    
    Args:
        probability: Probability of CKD (0-1)
    
    Returns:
        Risk category string: Low, Medium, or High
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


def get_risk_color(category: str) -> str:
    """Get color code for risk category."""
    colors = {
        "Low": "#28a745",      # Green
        "Medium": "#ffc107",   # Yellow
        "High": "#dc3545"      # Red
    }
    return colors.get(category, "#6c757d")


def get_kidney_function_interpretation(score: float) -> str:
    """
    Interpret kidney function score.
    
    Args:
        score: Kidney function score (typically 0-100 or normalized)
    
    Returns:
        Interpretation string
    """
    if score >= 90:
        return "Normal kidney function"
    elif score >= 60:
        return "Mildly decreased kidney function"
    elif score >= 30:
        return "Moderately decreased kidney function"
    elif score >= 15:
        return "Severely decreased kidney function"
    else:
        return "Kidney failure"


def validate_input_data(data: dict, required_fields: list) -> tuple:
    """
    Validate input data has all required fields.
    
    Args:
        data: Input data dictionary
        required_fields: List of required field names
    
    Returns:
        Tuple of (is_valid, missing_fields)
    """
    missing = [field for field in required_fields if field not in data or data[field] is None]
    return len(missing) == 0, missing
