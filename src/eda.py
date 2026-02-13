"""
Exploratory Data Analysis (EDA) Module for CKD Prediction System

This module provides comprehensive visualization and statistical analysis
of the Chronic Kidney Disease dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class CKDExploratoryAnalysis:
    """
    Exploratory Data Analysis class for CKD dataset.
    
    Provides methods for:
    - Statistical summaries
    - Distribution plots
    - Correlation analysis
    - Feature relationships
    - Target variable analysis
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize EDA with dataset.
        
        Args:
            data: Pandas DataFrame containing the CKD dataset
        """
        self.data = data.copy()
        self.numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove ID and target columns from feature lists
        exclude_cols = ['patient_id', 'kidney_function_score']
        self.numerical_features = [col for col in self.numerical_cols if col not in exclude_cols]
        self.categorical_features = [col for col in self.categorical_cols if col != 'ckd']
        
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics.
        
        Returns:
            DataFrame with summary statistics
        """
        summary = self.data.describe(include='all').T
        summary['missing'] = self.data.isnull().sum()
        summary['missing_pct'] = (self.data.isnull().sum() / len(self.data) * 100).round(2)
        summary['unique'] = self.data.nunique()
        
        return summary
    
    def plot_target_distribution(self, figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
        """
        Plot distribution of target variables (CKD and Kidney Function Score).
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # CKD distribution (Classification target)
        if 'ckd' in self.data.columns:
            ckd_counts = self.data['ckd'].value_counts()
            colors = ['#2ecc71', '#e74c3c']
            axes[0].pie(ckd_counts.values, labels=ckd_counts.index, autopct='%1.1f%%',
                       colors=colors, explode=(0.05, 0.05), shadow=True)
            axes[0].set_title('CKD Distribution\n(Classification Target)', fontsize=12, fontweight='bold')
        
        # Kidney Function Score distribution (Regression target)
        if 'kidney_function_score' in self.data.columns:
            axes[1].hist(self.data['kidney_function_score'], bins=30, color='#3498db', 
                        edgecolor='white', alpha=0.7)
            axes[1].axvline(self.data['kidney_function_score'].mean(), color='#e74c3c', 
                           linestyle='--', linewidth=2, label=f'Mean: {self.data["kidney_function_score"].mean():.1f}')
            axes[1].axvline(self.data['kidney_function_score'].median(), color='#2ecc71', 
                           linestyle='--', linewidth=2, label=f'Median: {self.data["kidney_function_score"].median():.1f}')
            axes[1].set_xlabel('Kidney Function Score', fontsize=10)
            axes[1].set_ylabel('Frequency', fontsize=10)
            axes[1].set_title('Kidney Function Score Distribution\n(Regression Target)', fontsize=12, fontweight='bold')
            axes[1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_numerical_distributions(self, figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Plot distributions of all numerical features.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        n_features = len(self.numerical_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(self.numerical_features):
            ax = axes[i]
            self.data[col].hist(ax=ax, bins=25, color='#3498db', edgecolor='white', alpha=0.7)
            ax.set_title(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Frequency')
            
            # Add mean line
            mean_val = self.data[col].mean()
            ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=1.5)
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Numerical Features Distribution', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_categorical_distributions(self, figsize: Tuple[int, int] = (16, 14)) -> plt.Figure:
        """
        Plot distributions of categorical features.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        n_features = len(self.categorical_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        colors = sns.color_palette("husl", 8)
        
        for i, col in enumerate(self.categorical_features):
            ax = axes[i]
            value_counts = self.data[col].value_counts()
            bars = ax.bar(range(len(value_counts)), value_counts.values, color=colors[:len(value_counts)])
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=8)
            ax.set_title(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_ylabel('Count')
            
            # Add count labels on bars
            for bar, count in zip(bars, value_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(count), ha='center', va='bottom', fontsize=8)
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Categorical Features Distribution', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, figsize: Tuple[int, int] = (14, 12)) -> plt.Figure:
        """
        Plot correlation matrix heatmap for numerical features.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        # Get numerical columns including targets
        numerical_data = self.data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numerical_data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, square=True,
                   linewidths=0.5, cbar_kws={'shrink': 0.8},
                   ax=ax, annot_kws={'size': 8})
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        return fig
    
    def plot_features_vs_target(self, target: str = 'ckd', figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Plot numerical features against target variable.
        
        Args:
            target: Target variable name ('ckd' or 'kidney_function_score')
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        n_features = len(self.numerical_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(self.numerical_features):
            ax = axes[i]
            
            if target == 'ckd':
                # Box plot for classification target
                self.data.boxplot(column=col, by=target, ax=ax)
                ax.set_title(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
                ax.set_xlabel('CKD Status')
            else:
                # Scatter plot for regression target
                ax.scatter(self.data[col], self.data[target], alpha=0.5, c='#3498db')
                ax.set_xlabel(col.replace('_', ' ').title())
                ax.set_ylabel('Kidney Function Score')
                ax.set_title(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(f'Numerical Features vs {target.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_categorical_vs_target(self, figsize: Tuple[int, int] = (16, 14)) -> plt.Figure:
        """
        Plot categorical features against CKD target.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        n_features = len(self.categorical_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(self.categorical_features):
            ax = axes[i]
            
            # Create crosstab
            ct = pd.crosstab(self.data[col], self.data['ckd'], normalize='index') * 100
            
            ct.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], width=0.8)
            ax.set_title(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Percentage')
            ax.legend(title='CKD', loc='upper right')
            ax.tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Categorical Features vs CKD Status', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_age_analysis(self, figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
        """
        Analyze age distribution by CKD status.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Age distribution by CKD status
        for status, color in zip(['No', 'Yes'], ['#2ecc71', '#e74c3c']):
            subset = self.data[self.data['ckd'] == status]['age']
            axes[0].hist(subset, bins=20, alpha=0.6, label=f'CKD: {status}', color=color)
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Age Distribution by CKD Status', fontweight='bold')
        axes[0].legend()
        
        # Age vs Kidney Function Score
        axes[1].scatter(self.data['age'], self.data['kidney_function_score'], 
                       c=self.data['ckd'].map({'Yes': 1, 'No': 0}), 
                       cmap='RdYlGn_r', alpha=0.6)
        axes[1].set_xlabel('Age')
        axes[1].set_ylabel('Kidney Function Score')
        axes[1].set_title('Age vs Kidney Function Score', fontweight='bold')
        
        # Age groups analysis
        self.data['age_group'] = pd.cut(self.data['age'], 
                                        bins=[0, 40, 55, 70, 100], 
                                        labels=['<40', '40-55', '55-70', '70+'])
        age_ckd = self.data.groupby('age_group')['ckd'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        )
        axes[2].bar(age_ckd.index.astype(str), age_ckd.values, color='#e74c3c', alpha=0.7)
        axes[2].set_xlabel('Age Group')
        axes[2].set_ylabel('CKD Percentage (%)')
        axes[2].set_title('CKD Rate by Age Group', fontweight='bold')
        
        # Add percentage labels
        for i, (idx, val) in enumerate(age_ckd.items()):
            axes[2].text(i, val + 1, f'{val:.1f}%', ha='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_risk_factors_analysis(self, figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        Analyze key risk factors for CKD.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        risk_factors = ['hypertension', 'diabetes_mellitus', 'smoking', 'family_history_ckd']
        
        fig, axes = plt.subplots(1, len(risk_factors), figsize=figsize)
        
        for i, factor in enumerate(risk_factors):
            # Calculate CKD rate for each risk factor status
            ct = pd.crosstab(self.data[factor], self.data['ckd'], normalize='index') * 100
            
            if 'Yes' in ct.columns:
                ckd_rates = ct['Yes']
            else:
                ckd_rates = pd.Series([0] * len(ct), index=ct.index)
            
            bars = axes[i].bar(ckd_rates.index, ckd_rates.values, 
                              color=['#2ecc71', '#e74c3c'] if len(ckd_rates) == 2 else ['#3498db'])
            axes[i].set_title(factor.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            axes[i].set_ylabel('CKD Rate (%)')
            axes[i].set_ylim(0, 100)
            
            # Add percentage labels
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 2,
                            f'{height:.1f}%', ha='center', fontsize=10)
        
        plt.suptitle('CKD Rate by Risk Factors', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def generate_eda_report(self, save_path: Optional[str] = None) -> dict:
        """
        Generate comprehensive EDA report with all visualizations.
        
        Args:
            save_path: Optional path to save figures
            
        Returns:
            Dictionary containing all figures and statistics
        """
        report = {
            'summary_statistics': self.get_summary_statistics(),
            'figures': {}
        }
        
        # Generate all plots
        report['figures']['target_distribution'] = self.plot_target_distribution()
        report['figures']['numerical_distributions'] = self.plot_numerical_distributions()
        report['figures']['categorical_distributions'] = self.plot_categorical_distributions()
        report['figures']['correlation_matrix'] = self.plot_correlation_matrix()
        report['figures']['features_vs_ckd'] = self.plot_features_vs_target('ckd')
        report['figures']['categorical_vs_target'] = self.plot_categorical_vs_target()
        report['figures']['age_analysis'] = self.plot_age_analysis()
        report['figures']['risk_factors'] = self.plot_risk_factors_analysis()
        
        # Save figures if path provided
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            for name, fig in report['figures'].items():
                fig.savefig(save_dir / f'{name}.png', dpi=150, bbox_inches='tight')
        
        return report


def run_eda(data_path: str, save_path: Optional[str] = None):
    """
    Run complete EDA on the CKD dataset.
    
    Args:
        data_path: Path to the CSV data file
        save_path: Optional path to save visualizations
    """
    # Load data
    data = pd.read_csv(data_path)
    print(f"Data loaded. Shape: {data.shape}")
    
    # Initialize EDA
    eda = CKDExploratoryAnalysis(data)
    
    # Generate report
    report = eda.generate_eda_report(save_path)
    
    # Print summary
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS SUMMARY")
    print("="*50)
    print(f"\nDataset Shape: {data.shape}")
    print(f"Numerical Features: {len(eda.numerical_features)}")
    print(f"Categorical Features: {len(eda.categorical_features)}")
    print(f"\nCKD Distribution:")
    print(data['ckd'].value_counts())
    print(f"\nKidney Function Score Statistics:")
    print(data['kidney_function_score'].describe())
    
    return report


if __name__ == "__main__":
    from pathlib import Path
    
    # Get data path
    data_path = Path(__file__).parent.parent / "data" / "Chronic_Kidney_Disease_Risk_Assessment.csv"
    
    # Run EDA
    report = run_eda(str(data_path))
    plt.show()
