"""
Utility functions for Housing Price Prediction Model
=====================================================
Additional helper functions for data preprocessing, visualization, and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles data preprocessing and cleaning."""
    
    @staticmethod
    def handle_missing_values(df, strategy='drop'):
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): 'drop' or 'mean' or 'median'
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        if strategy == 'drop':
            return df.dropna()
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            return df
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            return df
    
    @staticmethod
    def detect_outliers(df, column, method='iqr', threshold=1.5):
        """
        Detect outliers in a column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name
            method (str): 'iqr' or 'zscore'
            threshold (float): IQR multiplier or z-score threshold
            
        Returns:
            np.ndarray: Boolean array of outlier indices
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column]))
            return z_scores > threshold
    
    @staticmethod
    def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
        """
        Remove outliers from specified columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): Columns to check for outliers
            method (str): 'iqr' or 'zscore'
            threshold (float): Threshold value
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outlier_mask = pd.Series([False] * len(df))
        for col in columns:
            outlier_mask |= DataPreprocessor.detect_outliers(
                df, col, method=method, threshold=threshold
            )
        
        return df[~outlier_mask]
    
    @staticmethod
    def normalize_features(df, columns=None):
        """
        Normalize features to 0-1 range.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): Columns to normalize
            
        Returns:
            pd.DataFrame: Normalized dataframe
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df_normalized = df.copy()
        for col in columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        return df_normalized


class ModelEvaluator:
    """Advanced model evaluation and comparison."""
    
    @staticmethod
    def evaluate_with_cv(model, X, y, cv=5):
        """
        Evaluate model with cross-validation.
        
        Args:
            model: Scikit-learn model
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Number of folds
            
        Returns:
            dict: CV scores and statistics
        """
        cv_scores = cross_val_score(model, X, y, cv=cv, 
                                    scoring='r2', n_jobs=-1)
        
        return {
            'mean_r2': cv_scores.mean(),
            'std_r2': cv_scores.std(),
            'scores': cv_scores,
            'rmse': np.sqrt(-cross_val_score(model, X, y, cv=cv,
                                            scoring='neg_mean_squared_error')).mean()
        }
    
    @staticmethod
    def hyperparameter_tuning(model, param_grid, X, y, cv=5):
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            model: Scikit-learn model
            param_grid (dict): Parameter grid
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Number of folds
            
        Returns:
            GridSearchCV: Fitted grid search object
        """
        grid_search = GridSearchCV(model, param_grid, cv=cv, 
                                  n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    @staticmethod
    def residual_analysis(y_true, y_pred):
        """
        Analyze model residuals.
        
        Args:
            y_true (pd.Series): Actual values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            dict: Residual statistics
        """
        residuals = y_true - y_pred
        
        return {
            'mean_residual': residuals.mean(),
            'std_residual': residuals.std(),
            'min_residual': residuals.min(),
            'max_residual': residuals.max(),
            'residuals': residuals
        }


class Visualizer:
    """Advanced visualization utilities."""
    
    @staticmethod
    def plot_residuals(y_true, y_pred, title="Residual Analysis"):
        """
        Create residual diagnostic plots.
        
        Args:
            y_true (pd.Series): Actual values
            y_pred (np.ndarray): Predicted values
            title (str): Plot title
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scale-location plot
        standardized_residuals = residuals / residuals.std()
        axes[1, 1].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('√|Standardized Residuals|')
        axes[1, 1].set_title('Scale-Location Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig(f'/home/claude/residual_analysis_{title.replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_feature_correlation(df, target_col=None, figsize=(10, 8)):
        """
        Create a correlation heatmap.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column for sorting
            figsize (tuple): Figure size
        """
        corr_matrix = df.corr()
        
        if target_col and target_col in df.columns:
            corr_with_target = corr_matrix[target_col].sort_values(ascending=False)
            corr_matrix = corr_matrix.loc[corr_with_target.index, 
                                         corr_with_target.index]
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('/home/claude/detailed_correlation_matrix.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_learning_curves(train_scores, val_scores, xlabel='Training Set Size'):
        """
        Plot learning curves.
        
        Args:
            train_scores (list): Training scores
            val_scores (list): Validation scores
            xlabel (str): X-axis label
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_range = range(1, len(train_scores) + 1)
        
        ax.plot(x_range, train_scores, marker='o', label='Training Score')
        ax.plot(x_range, val_scores, marker='s', label='Validation Score')
        ax.fill_between(x_range, train_scores, val_scores, alpha=0.1)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel('R² Score')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/claude/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_error_distribution(y_true, y_pred, bins=30):
        """
        Plot prediction error distribution.
        
        Args:
            y_true (pd.Series): Actual values
            y_pred (np.ndarray): Predicted values
            bins (int): Number of bins in histogram
        """
        errors = np.abs(y_true - y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Error histogram
        axes[0].hist(errors, bins=bins, edgecolor='black', alpha=0.7)
        axes[0].axvline(errors.mean(), color='r', linestyle='--', 
                       label=f'Mean: ${errors.mean():,.0f}')
        axes[0].set_xlabel('Absolute Error ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Prediction Errors')
        axes[0].legend()
        
        # Percentage error
        pct_errors = (errors / y_true) * 100
        axes[1].hist(pct_errors, bins=bins, edgecolor='black', alpha=0.7, color='orange')
        axes[1].axvline(pct_errors.mean(), color='r', linestyle='--',
                       label=f'Mean: {pct_errors.mean():.2f}%')
        axes[1].set_xlabel('Percentage Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Percentage Errors')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('/home/claude/error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()


class MetricsCalculator:
    """Calculate additional performance metrics."""
    
    @staticmethod
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error."""
        return mean_absolute_percentage_error(y_true, y_pred) * 100
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def rmsle(y_true, y_pred):
        """Root Mean Squared Logarithmic Error."""
        return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
    
    @staticmethod
    def median_absolute_error(y_true, y_pred):
        """Median Absolute Error."""
        return np.median(np.abs(y_true - y_pred))
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred):
        """
        Calculate all available metrics.
        
        Args:
            y_true (pd.Series): Actual values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            dict: All calculated metrics
        """
        from sklearn.metrics import r2_score, mean_absolute_error
        
        return {
            'R² Score': r2_score(y_true, y_pred),
            'RMSE': MetricsCalculator.rmse(y_true, y_pred),
            'RMSLE': MetricsCalculator.rmsle(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'Median AE': MetricsCalculator.median_absolute_error(y_true, y_pred),
            'MAPE': MetricsCalculator.mape(y_true, y_pred)
        }


# Example usage functions
def example_preprocessing():
    """Example: Data preprocessing."""
    print("=" * 60)
    print("DATA PREPROCESSING EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 100, 5],  # Contains outlier
        'feature2': [10, 20, 30, 40, 50],
        'target': [100, 200, 300, 400, 500]
    })
    
    print("\nOriginal data:")
    print(df)
    
    # Remove outliers
    df_clean = DataPreprocessor.remove_outliers(df, columns=['feature1'])
    print("\nAfter removing outliers:")
    print(df_clean)
    
    # Normalize
    df_normalized = DataPreprocessor.normalize_features(df_clean)
    print("\nAfter normalization:")
    print(df_normalized)


if __name__ == "__main__":
    example_preprocessing()
