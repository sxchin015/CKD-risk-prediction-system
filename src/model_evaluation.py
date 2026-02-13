"""
Model Evaluation Module for CKD Prediction System

This module provides comprehensive evaluation and visualization
for both classification and regression models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization class.
    
    Provides:
    - Classification metrics visualization
    - Regression metrics visualization
    - Model comparison charts
    - Performance reports
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.classification_results = {}
        self.regression_results = {}
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        labels: List[str] = ['No CKD', 'CKD'],
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            labels: Class labels for display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   ax=ax, annot_kws={'size': 14})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        # Add metrics below the matrix
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics_text = f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}'
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(
        self,
        results: Dict[str, Dict],
        y_test: np.ndarray,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple models.
        
        Args:
            results: Dictionary of model results containing y_prob
            y_test: True labels
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        for (model_name, metrics), color in zip(results.items(), colors):
            if 'y_prob' in metrics:
                fpr, tpr, _ = roc_curve(y_test, metrics['y_prob'])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color=color, lw=2,
                       label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curves(
        self,
        results: Dict[str, Dict],
        y_test: np.ndarray,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            results: Dictionary of model results containing y_prob
            y_test: True labels
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        for (model_name, metrics), color in zip(results.items(), colors):
            if 'y_prob' in metrics:
                precision, recall, _ = precision_recall_curve(y_test, metrics['y_prob'])
                
                ax.plot(recall, precision, color=color, lw=2,
                       label=f'{model_name.replace("_", " ").title()}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_classification_comparison(
        self,
        comparison_df: pd.DataFrame,
        figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Plot classification model comparison bar charts.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Metrics bar chart (support both % and 0-1 formats)
        metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'ROC-AUC (%)',
                   'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        x = np.arange(len(comparison_df))
        width = 0.15
        
        for i, metric in enumerate(available_metrics):
            axes[0].bar(x + i * width, comparison_df[metric], width, 
                       label=metric, alpha=0.8)
        
        axes[0].set_xlabel('Model', fontsize=11)
        axes[0].set_ylabel('Score', fontsize=11)
        axes[0].set_title('Classification Metrics Comparison', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x + width * 2)
        axes[0].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        axes[0].legend(loc='lower right', fontsize=9)
        y_max = comparison_df[available_metrics].max().max()
        axes[0].set_ylim(0, y_max * 1.1 if y_max > 1 else 1.1)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # ROC-AUC horizontal bar chart
        roc_col = 'ROC-AUC (%)' if 'ROC-AUC (%)' in comparison_df.columns else 'ROC-AUC'
        acc_col = 'Accuracy (%)' if 'Accuracy (%)' in comparison_df.columns else 'Accuracy'
        roc_vals = comparison_df[roc_col]
        colors = plt.cm.RdYlGn(roc_vals / 100 if roc_vals.max() > 1 else roc_vals)
        bars = axes[1].barh(comparison_df['Model'], roc_vals, color=colors, alpha=0.8)
        
        axes[1].set_xlabel('ROC-AUC Score (%)', fontsize=11)
        axes[1].set_title('Model Ranking by ROC-AUC', fontsize=12, fontweight='bold')
        axes[1].set_xlim(0, 110 if roc_vals.max() > 1 else 1.1)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            w = bar.get_width()
            axes[1].text(w + 1, bar.get_y() + bar.get_height()/2,
                        f'{w:.1f}%' if w > 1 else f'{w:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_regression_comparison(
        self,
        comparison_df: pd.DataFrame,
        figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Plot regression model comparison charts.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # R² bar chart
        colors = plt.cm.RdYlGn(comparison_df['R²'])
        bars = axes[0].barh(comparison_df['Model'], comparison_df['R²'],
                           color=colors, alpha=0.8)
        
        axes[0].set_xlabel('R² Score', fontsize=11)
        axes[0].set_title('Model Ranking by R²', fontsize=12, fontweight='bold')
        axes[0].set_xlim(0, 1.1)
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            axes[0].text(width + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', va='center', fontsize=10)
        
        # Error metrics comparison
        x = np.arange(len(comparison_df))
        width = 0.35
        
        axes[1].bar(x - width/2, comparison_df['RMSE'], width, label='RMSE', color='#e74c3c', alpha=0.8)
        axes[1].bar(x + width/2, comparison_df['MAE'], width, label='MAE', color='#3498db', alpha=0.8)
        
        axes[1].set_xlabel('Model', fontsize=11)
        axes[1].set_ylabel('Error', fontsize=11)
        axes[1].set_title('Error Metrics Comparison', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        figsize: Tuple[int, int] = (14, 5)
    ) -> plt.Figure:
        """
        Plot residual analysis for regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, c='#3498db')
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values', fontsize=11)
        axes[0].set_ylabel('Residuals', fontsize=11)
        axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1].hist(residuals, bins=30, color='#3498db', edgecolor='white', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Actual vs Predicted
        axes[2].scatter(y_true, y_pred, alpha=0.6, c='#3498db')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        axes[2].set_xlabel('Actual Values', fontsize=11)
        axes[2].set_ylabel('Predicted Values', fontsize=11)
        axes[2].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle(f'Residual Analysis - {model_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        model_name: str,
        top_n: int = 15,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            importance_df: DataFrame with feature importances
            model_name: Name of the model
            top_n: Number of top features to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Get top N features
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'],
                      color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].apply(lambda x: x.replace('_', ' ').title()))
        ax.invert_yaxis()  # Top feature at top
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def generate_evaluation_report(
        self,
        classification_results: Dict,
        regression_results: Dict,
        y_test_cls: np.ndarray,
        y_test_reg: np.ndarray,
        classification_comparison: pd.DataFrame,
        regression_comparison: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive evaluation report with all visualizations.
        
        Args:
            classification_results: Classification model results
            regression_results: Regression model results
            y_test_cls: Test labels for classification
            y_test_reg: Test values for regression
            classification_comparison: Classification comparison DataFrame
            regression_comparison: Regression comparison DataFrame
            save_path: Optional path to save figures
            
        Returns:
            Dictionary containing all figures
        """
        report = {'figures': {}}
        
        # Classification plots
        report['figures']['roc_curves'] = self.plot_roc_curves(classification_results, y_test_cls)
        report['figures']['pr_curves'] = self.plot_precision_recall_curves(classification_results, y_test_cls)
        report['figures']['classification_comparison'] = self.plot_classification_comparison(classification_comparison)
        
        # Confusion matrices for each classification model
        for model_name, metrics in classification_results.items():
            if 'y_pred' in metrics:
                fig = self.plot_confusion_matrix(
                    y_test_cls, metrics['y_pred'],
                    model_name.replace('_', ' ').title()
                )
                report['figures'][f'confusion_matrix_{model_name}'] = fig
        
        # Regression plots
        report['figures']['regression_comparison'] = self.plot_regression_comparison(regression_comparison)
        
        # Residual plots for each regression model
        for model_name, metrics in regression_results.items():
            if 'y_pred' in metrics:
                fig = self.plot_residuals(
                    metrics['y_test'], metrics['y_pred'],
                    model_name.replace('_', ' ').title()
                )
                report['figures'][f'residuals_{model_name}'] = fig
        
        # Save figures if path provided
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            for name, fig in report['figures'].items():
                fig.savefig(save_dir / f'{name}.png', dpi=150, bbox_inches='tight')
        
        return report


def create_summary_dashboard(
    classification_comparison: pd.DataFrame,
    regression_comparison: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Create a summary dashboard with all key metrics.
    
    Args:
        classification_comparison: Classification comparison DataFrame
        regression_comparison: Regression comparison DataFrame
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Classification ROC-AUC
    roc_col = 'ROC-AUC (%)' if 'ROC-AUC (%)' in classification_comparison.columns else 'ROC-AUC'
    roc_vals = classification_comparison[roc_col]
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.RdYlGn(roc_vals / 100 if roc_vals.max() > 1 else roc_vals)
    bars = ax1.barh(classification_comparison['Model'], roc_vals, color=colors, alpha=0.8)
    ax1.set_xlabel('ROC-AUC (%)', fontsize=11)
    ax1.set_title('Classification: ROC-AUC Scores', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 110 if roc_vals.max() > 1 else 1.1)
    ax1.grid(True, alpha=0.3, axis='x')
    for bar in bars:
        w = bar.get_width()
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{w:.1f}%' if w > 1 else f'{w:.3f}', va='center', fontsize=9)
    
    # Classification F1
    f1_col = 'F1-Score (%)' if 'F1-Score (%)' in classification_comparison.columns else 'F1-Score'
    f1_vals = classification_comparison[f1_col]
    ax2 = fig.add_subplot(gs[0, 1])
    colors = plt.cm.RdYlGn(f1_vals / 100 if f1_vals.max() > 1 else f1_vals)
    bars = ax2.barh(classification_comparison['Model'], f1_vals, color=colors, alpha=0.8)
    ax2.set_xlabel('F1-Score (%)', fontsize=11)
    ax2.set_title('Classification: F1 Scores', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 110 if f1_vals.max() > 1 else 1.1)
    ax2.grid(True, alpha=0.3, axis='x')
    for bar in bars:
        w = bar.get_width()
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{w:.1f}%' if w > 1 else f'{w:.3f}', va='center', fontsize=9)
    
    # Regression R²
    ax3 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.RdYlGn(regression_comparison['R²'])
    bars = ax3.barh(regression_comparison['Model'], regression_comparison['R²'],
                   color=colors, alpha=0.8)
    ax3.set_xlabel('R²', fontsize=11)
    ax3.set_title('Regression: R² Scores', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 1.1)
    ax3.grid(True, alpha=0.3, axis='x')
    for bar in bars:
        ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.3f}', va='center', fontsize=9)
    
    # Regression RMSE
    ax4 = fig.add_subplot(gs[1, 1])
    colors = plt.cm.RdYlGn_r(regression_comparison['RMSE'] / regression_comparison['RMSE'].max())
    bars = ax4.barh(regression_comparison['Model'], regression_comparison['RMSE'],
                   color=colors, alpha=0.8)
    ax4.set_xlabel('RMSE', fontsize=11)
    ax4.set_title('Regression: RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    for bar in bars:
        ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.2f}', va='center', fontsize=9)
    
    fig.suptitle('CKD Prediction - Model Performance Summary', fontsize=16, fontweight='bold', y=1.02)
    
    return fig


if __name__ == "__main__":
    # Example usage
    from data_pipeline import CKDDataPipeline
    from classification_models import CKDClassificationModels
    from regression_models import CKDRegressionModels
    
    # Load data
    pipeline = CKDDataPipeline()
    
    # Classification data
    X_train_cls, X_test_cls, y_train_cls, y_test_cls, features = pipeline.get_classification_data()
    
    # Train and evaluate classification models
    classifier = CKDClassificationModels()
    classifier.train_all_models(X_train_cls, y_train_cls, feature_names=features)
    classifier.evaluate_all_models(X_test_cls, y_test_cls)
    cls_comparison = classifier.get_comparison_table()
    
    # Regression data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, features = pipeline.get_regression_data()
    
    # Train and evaluate regression models
    regressor = CKDRegressionModels()
    regressor.train_all_models(X_train_reg, y_train_reg, feature_names=features)
    regressor.evaluate_all_models(X_test_reg, y_test_reg)
    reg_comparison = regressor.get_comparison_table()
    
    # Create evaluator and generate report
    evaluator = ModelEvaluator()
    report = evaluator.generate_evaluation_report(
        classifier.results,
        regressor.results,
        y_test_cls,
        y_test_reg,
        cls_comparison,
        reg_comparison
    )
    
    # Create summary dashboard
    dashboard = create_summary_dashboard(cls_comparison, reg_comparison)
    
    plt.show()
