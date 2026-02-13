"""
Model Explainability Module for CKD Prediction System

This module provides interpretability and explainability features including:
- SHAP values analysis
- Feature importance visualization
- Individual prediction explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CKDExplainer:
    """
    Model explainability class for CKD prediction models.
    
    Provides:
    - SHAP-based explanations
    - Feature importance analysis
    - Local (individual) explanations
    - Global model behavior insights
    """
    
    def __init__(self, model: Any, feature_names: List[str], model_type: str = 'classification'):
        """
        Initialize the explainer.
        
        Args:
            model: Trained model object
            feature_names: List of feature names
            model_type: 'classification' or 'regression'
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.X_background = None
    
    def setup_shap_explainer(self, X_background: np.ndarray, max_samples: int = 100) -> None:
        """
        Set up SHAP explainer with background data.
        
        Args:
            X_background: Background data for SHAP
            max_samples: Maximum number of background samples to use
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping explainer setup.")
            return
        
        # Limit background samples for computational efficiency
        if len(X_background) > max_samples:
            indices = np.random.choice(len(X_background), max_samples, replace=False)
            X_background = X_background[indices]
        
        self.X_background = X_background
        
        try:
            # Try TreeExplainer first (for tree-based models)
            if hasattr(self.model, 'feature_importances_'):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Fall back to KernelExplainer
                if self.model_type == 'classification':
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        X_background
                    )
                else:
                    self.explainer = shap.KernelExplainer(
                        self.model.predict, 
                        X_background
                    )
            
            logger.info(f"SHAP explainer set up successfully: {type(self.explainer).__name__}")
        except Exception as e:
            logger.error(f"Error setting up SHAP explainer: {e}")
            self.explainer = None
    
    def calculate_shap_values(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate SHAP values for given data.
        
        Args:
            X: Data to explain
            
        Returns:
            SHAP values array or None if not available
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return None
        
        try:
            self.shap_values = self.explainer.shap_values(X)
            
            # Handle different SHAP value formats
            if isinstance(self.shap_values, list):
                # For multi-class classification, take values for positive class
                self.shap_values = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
            
            return self.shap_values
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return None
    
    def get_feature_importance_from_shap(self, X: np.ndarray) -> pd.DataFrame:
        """
        Calculate feature importance from SHAP values.
        
        Args:
            X: Data to calculate importance for
            
        Returns:
            DataFrame with feature importances
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        if self.shap_values is None:
            return self._get_model_feature_importance()
        
        # Calculate mean absolute SHAP value for each feature
        importance = np.abs(self.shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def _get_model_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance directly from model if SHAP is not available.
        
        Returns:
            DataFrame with feature importances
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_).flatten()
        else:
            # Return equal importance if not available
            importance = np.ones(len(self.feature_names)) / len(self.feature_names)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def explain_single_prediction(
        self,
        X_instance: np.ndarray,
        prediction: float,
        top_n: int = 10
    ) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            X_instance: Single instance features (1D or 2D array)
            prediction: Model prediction for this instance
            top_n: Number of top features to include
            
        Returns:
            Dictionary with explanation details
        """
        # Ensure 2D array
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        explanation = {
            'prediction': prediction,
            'features': {},
            'top_positive_factors': [],
            'top_negative_factors': [],
            'shap_available': False
        }
        
        # Get SHAP values for this instance
        shap_vals = None
        if SHAP_AVAILABLE and self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(X_instance)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                shap_vals = shap_vals.flatten()
                explanation['shap_available'] = True
            except Exception as e:
                logger.warning(f"Could not calculate SHAP values: {e}")
        
        # Create feature explanations
        feature_values = X_instance.flatten()
        
        for i, (name, value) in enumerate(zip(self.feature_names, feature_values)):
            feature_info = {
                'value': value,
                'shap_value': float(shap_vals[i]) if shap_vals is not None else None
            }
            explanation['features'][name] = feature_info
        
        # Get top positive and negative factors
        if shap_vals is not None:
            feature_shap = list(zip(self.feature_names, shap_vals))
            feature_shap_sorted = sorted(feature_shap, key=lambda x: x[1], reverse=True)
            
            # Top factors increasing prediction
            explanation['top_positive_factors'] = [
                {'feature': f, 'shap_value': float(s)}
                for f, s in feature_shap_sorted[:top_n] if s > 0
            ]
            
            # Top factors decreasing prediction
            explanation['top_negative_factors'] = [
                {'feature': f, 'shap_value': float(s)}
                for f, s in feature_shap_sorted[-top_n:][::-1] if s < 0
            ]
        
        return explanation
    
    def plot_shap_summary(
        self,
        X: np.ndarray,
        max_display: int = 15,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Optional[plt.Figure]:
        """
        Create SHAP summary plot.
        
        Args:
            X: Data to create summary for
            max_display: Maximum number of features to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure or None if SHAP not available
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            logger.warning("SHAP not available. Cannot create summary plot.")
            return None
        
        # Calculate SHAP values if not already done
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        if self.shap_values is None:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create summary plot
        shap.summary_plot(
            self.shap_values, X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_feature_importance(
        self,
        X: np.ndarray,
        top_n: int = 15,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            X: Data for importance calculation
            top_n: Number of top features to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Get feature importance
        importance_df = self.get_feature_importance_from_shap(X)
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'],
                      color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].apply(lambda x: x.replace('_', ' ').title()))
        ax.invert_yaxis()
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_single_explanation(
        self,
        explanation: Dict,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Visualize a single prediction explanation.
        
        Args:
            explanation: Explanation dictionary from explain_single_prediction
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Positive factors
        if explanation['top_positive_factors']:
            factors = explanation['top_positive_factors'][:8]
            features = [f['feature'].replace('_', ' ').title() for f in factors]
            values = [f['shap_value'] for f in factors]
            
            axes[0].barh(range(len(features)), values, color='#e74c3c', alpha=0.8)
            axes[0].set_yticks(range(len(features)))
            axes[0].set_yticklabels(features)
            axes[0].invert_yaxis()
            axes[0].set_xlabel('SHAP Value (Impact on Prediction)')
            axes[0].set_title('Factors Increasing CKD Risk', fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3, axis='x')
        else:
            axes[0].text(0.5, 0.5, 'No positive factors', ha='center', va='center')
            axes[0].set_title('Factors Increasing CKD Risk', fontsize=12, fontweight='bold')
        
        # Negative factors
        if explanation['top_negative_factors']:
            factors = explanation['top_negative_factors'][:8]
            features = [f['feature'].replace('_', ' ').title() for f in factors]
            values = [abs(f['shap_value']) for f in factors]
            
            axes[1].barh(range(len(features)), values, color='#2ecc71', alpha=0.8)
            axes[1].set_yticks(range(len(features)))
            axes[1].set_yticklabels(features)
            axes[1].invert_yaxis()
            axes[1].set_xlabel('SHAP Value (Impact on Prediction)')
            axes[1].set_title('Factors Decreasing CKD Risk', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='x')
        else:
            axes[1].text(0.5, 0.5, 'No negative factors', ha='center', va='center')
            axes[1].set_title('Factors Decreasing CKD Risk', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Prediction Explanation (Prediction: {explanation["prediction"]:.3f})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def generate_text_explanation(
        self,
        explanation: Dict,
        patient_data: Dict
    ) -> str:
        """
        Generate a text-based explanation of the prediction.
        
        Args:
            explanation: Explanation dictionary
            patient_data: Original patient data dictionary
            
        Returns:
            Human-readable explanation string
        """
        pred = explanation['prediction']
        
        # Determine risk level
        if pred < 0.3:
            risk_level = "LOW"
            risk_msg = "The patient is at low risk for Chronic Kidney Disease."
        elif pred < 0.7:
            risk_level = "MODERATE"
            risk_msg = "The patient is at moderate risk for Chronic Kidney Disease and should be monitored."
        else:
            risk_level = "HIGH"
            risk_msg = "The patient is at high risk for Chronic Kidney Disease and requires immediate attention."
        
        text = f"""
╔══════════════════════════════════════════════════════════════╗
║                    CKD RISK ASSESSMENT                       ║
╠══════════════════════════════════════════════════════════════╣
║  Risk Probability: {pred*100:5.1f}%                                    ║
║  Risk Level: {risk_level:8}                                        ║
╚══════════════════════════════════════════════════════════════╝

{risk_msg}

"""
        
        # Add key risk factors
        if explanation['top_positive_factors']:
            text += "KEY RISK FACTORS (Increasing CKD Risk):\n"
            text += "─" * 45 + "\n"
            for factor in explanation['top_positive_factors'][:5]:
                feature = factor['feature'].replace('_', ' ').title()
                text += f"  ⚠ {feature}\n"
            text += "\n"
        
        # Add protective factors
        if explanation['top_negative_factors']:
            text += "PROTECTIVE FACTORS (Decreasing CKD Risk):\n"
            text += "─" * 45 + "\n"
            for factor in explanation['top_negative_factors'][:5]:
                feature = factor['feature'].replace('_', ' ').title()
                text += f"  ✓ {feature}\n"
            text += "\n"
        
        # Add recommendations based on risk factors
        text += "RECOMMENDATIONS:\n"
        text += "─" * 45 + "\n"
        
        if 'hypertension' in patient_data and patient_data.get('hypertension') == 'Yes':
            text += "  • Monitor and control blood pressure regularly\n"
        
        if 'diabetes_mellitus' in patient_data and patient_data.get('diabetes_mellitus') == 'Yes':
            text += "  • Manage blood glucose levels carefully\n"
        
        if 'smoking' in patient_data and patient_data.get('smoking') == 'Yes':
            text += "  • Consider smoking cessation program\n"
        
        text += "  • Regular kidney function tests recommended\n"
        text += "  • Maintain healthy diet and hydration\n"
        text += "  • Consult with nephrologist for detailed evaluation\n"
        
        return text


def create_explainability_report(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    model_type: str = 'classification',
    save_path: Optional[str] = None
) -> Dict:
    """
    Create comprehensive explainability report.
    
    Args:
        model: Trained model
        X_train: Training data
        X_test: Test data
        feature_names: List of feature names
        model_type: 'classification' or 'regression'
        save_path: Optional path to save figures
        
    Returns:
        Dictionary with explainability results
    """
    explainer = CKDExplainer(model, feature_names, model_type)
    
    # Set up SHAP explainer
    explainer.setup_shap_explainer(X_train)
    
    # Calculate SHAP values
    shap_values = explainer.calculate_shap_values(X_test)
    
    # Get feature importance
    importance_df = explainer.get_feature_importance_from_shap(X_test)
    
    report = {
        'feature_importance': importance_df,
        'explainer': explainer,
        'figures': {}
    }
    
    # Create visualizations
    report['figures']['feature_importance'] = explainer.plot_feature_importance(X_test)
    
    if SHAP_AVAILABLE:
        shap_fig = explainer.plot_shap_summary(X_test)
        if shap_fig:
            report['figures']['shap_summary'] = shap_fig
    
    # Save if path provided
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        importance_df.to_csv(save_dir / 'feature_importance.csv', index=False)
        
        for name, fig in report['figures'].items():
            fig.savefig(save_dir / f'{name}.png', dpi=150, bbox_inches='tight')
    
    return report


if __name__ == "__main__":
    # Example usage
    from data_pipeline import CKDDataPipeline
    from classification_models import CKDClassificationModels
    
    # Load data
    pipeline = CKDDataPipeline()
    X_train, X_test, y_train, y_test, feature_names = pipeline.get_classification_data()
    
    # Train a model
    classifier = CKDClassificationModels()
    classifier.train_model('random_forest', X_train, y_train, feature_names)
    model = classifier.trained_models['random_forest']
    
    # Create explainability report
    report = create_explainability_report(
        model, X_train, X_test, feature_names,
        model_type='classification'
    )
    
    print("Feature Importance:")
    print(report['feature_importance'].head(10))
    
    plt.show()
