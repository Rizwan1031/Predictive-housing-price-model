"""
Predictive Housing Price Model
================================
This project builds a machine learning model to predict house prices
using scikit-learn, pandas, and other data science libraries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class HousingPricePredictor:
    """
    A class to build and evaluate housing price prediction models.
    """
    
    def __init__(self, random_state=42):
        """Initialize the predictor with a random state for reproducibility."""
        self.random_state = random_state
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath=None):
        """
        Load housing data from CSV file or generate sample data.
        
        Args:
            filepath (str): Path to CSV file. If None, generates sample data.
        """
        if filepath:
            self.df = pd.read_csv(filepath)
            print(f"Data loaded from {filepath}")
        else:
            # Generate sample housing dataset
            np.random.seed(self.random_state)
            n_samples = 500
            
            self.df = pd.DataFrame({
                'square_feet': np.random.uniform(800, 5000, n_samples),
                'bedrooms': np.random.randint(1, 6, n_samples),
                'bathrooms': np.random.uniform(1, 4, n_samples),
                'age_years': np.random.uniform(0, 100, n_samples),
                'garage_spaces': np.random.randint(0, 4, n_samples),
                'lot_size': np.random.uniform(2000, 15000, n_samples),
                'condition': np.random.randint(1, 6, n_samples),  # 1-5 scale
                'location_quality': np.random.randint(1, 10, n_samples),  # 1-10 scale
            })
            
            # Generate target variable (price) with realistic relationships
            self.df['price'] = (
                self.df['square_feet'] * 150 +
                self.df['bedrooms'] * 30000 +
                self.df['bathrooms'] * 40000 -
                self.df['age_years'] * 500 +
                self.df['garage_spaces'] * 20000 +
                self.df['lot_size'] * 2 +
                self.df['condition'] * 25000 +
                self.df['location_quality'] * 40000 +
                np.random.normal(0, 50000, n_samples)  # Add noise
            )
            
            print(f"Sample dataset generated with {n_samples} records")
        
        print(f"\nDataset shape: {self.df.shape}")
        print(f"\nFirst few rows:\n{self.df.head()}")
        print(f"\nData info:\n{self.df.info()}")
        print(f"\nBasic statistics:\n{self.df.describe()}")
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Check for missing values
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        
        # Correlation analysis
        print(f"\nCorrelation with Price:")
        correlation_with_price = self.df.corr()['price'].sort_values(ascending=False)
        print(correlation_with_price)
        
        # Visualize correlations
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Housing Features')
        plt.tight_layout()
        plt.savefig(r"C:\Users\Admin\OneDrive\Documents\predictive housing price model\correlation_matrix.png", dpi=300, bbox_inches='tight')
        print("\n✓ Correlation matrix saved as 'correlation_matrix.png'")
        plt.close()
        
        # Price distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.df['price'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.title('Distribution of House Prices')
        
        plt.subplot(1, 2, 2)
        plt.scatter(self.df['square_feet'], self.df['price'], alpha=0.5)
        plt.xlabel('Square Feet')
        plt.ylabel('Price')
        plt.title('Price vs Square Footage')
        
        plt.tight_layout()
        plt.savefig(r"C:\Users\Admin\OneDrive\Documents\predictive housing price model\price_analysis.png", dpi=300, bbox_inches='tight')
        print("✓ Price analysis saved as 'price_analysis.png'")
        plt.close()
    
    def preprocess_data(self):
        """Preprocess the data for modeling."""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Handle missing values if any
        self.df = self.df.dropna()
        print(f"Dataset shape after removing NaN: {self.df.shape}")
        
        # Separate features and target
        X = self.df.drop('price', axis=1)
        y = self.df['price']
        
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        print(f"\nTrain set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("\n✓ Features scaled using StandardScaler")
    
    def train_models(self):
        """Train multiple regression models."""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Model 1: Linear Regression
        print("\n1. Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(self.X_train_scaled, self.y_train)
        self.models['Linear Regression'] = lr_model
        print("✓ Linear Regression trained")
        
        # Model 2: Random Forest
        print("2. Training Random Forest Regressor...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        print("✓ Random Forest trained")
        
        # Model 3: Gradient Boosting
        print("3. Training Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.random_state
        )
        gb_model.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb_model
        print("✓ Gradient Boosting trained")
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        results = []
        
        for model_name, model in self.models.items():
            # Make predictions
            if model_name == 'Linear Regression':
                y_pred_train = model.predict(self.X_train_scaled)
                y_pred_test = model.predict(self.X_test_scaled)
            else:
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            
            print(f"\n{model_name}:")
            print(f"  Train R² Score: {train_r2:.4f}")
            print(f"  Test R² Score:  {test_r2:.4f}")
            print(f"  Train RMSE:     ${train_rmse:,.2f}")
            print(f"  Test RMSE:      ${test_rmse:,.2f}")
            print(f"  Train MAE:      ${train_mae:,.2f}")
            print(f"  Test MAE:       ${test_mae:,.2f}")
            
            results.append({
                'Model': model_name,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Test RMSE': test_rmse,
                'Test MAE': test_mae
            })
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def feature_importance(self):
        """Display and plot feature importance."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Random Forest Feature Importance
        rf_model = self.models['Random Forest']
        rf_importance = pd.Series(
            rf_model.feature_importances_,
            index=self.X_train.columns
        ).sort_values(ascending=False)
        
        print("\nRandom Forest Feature Importance:")
        print(rf_importance)
        
        axes[0].barh(range(len(rf_importance)), rf_importance.values)
        axes[0].set_yticks(range(len(rf_importance)))
        axes[0].set_yticklabels(rf_importance.index)
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Random Forest - Feature Importance')
        axes[0].invert_yaxis()
        
        # Gradient Boosting Feature Importance
        gb_model = self.models['Gradient Boosting']
        gb_importance = pd.Series(
            gb_model.feature_importances_,
            index=self.X_train.columns
        ).sort_values(ascending=False)
        
        print("\nGradient Boosting Feature Importance:")
        print(gb_importance)
        
        axes[1].barh(range(len(gb_importance)), gb_importance.values, color='orange')
        axes[1].set_yticks(range(len(gb_importance)))
        axes[1].set_yticklabels(gb_importance.index)
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Gradient Boosting - Feature Importance')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(r"C:\Users\Admin\OneDrive\Documents\predictive housing price model\feature_importance.png", dpi=300, bbox_inches='tight')
        print("\n✓ Feature importance plot saved as 'feature_importance.png'")
        plt.close()
    
    def plot_predictions(self):
        """Plot actual vs predicted values."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        
        model_list = [
            ('Linear Regression', self.X_train_scaled, self.X_test_scaled),
            ('Random Forest', self.X_train, self.X_test),
            ('Gradient Boosting', self.X_train, self.X_test)
        ]
        
        for idx, (model_name, X_train, X_test) in enumerate(model_list):
            model = self.models[model_name]
            y_pred = model.predict(X_test)
            r2 = r2_score(self.y_test, y_pred)
            
            axes[idx].scatter(self.y_test, y_pred, alpha=0.5)
            axes[idx].plot([self.y_test.min(), self.y_test.max()],
                          [self.y_test.min(), self.y_test.max()],
                          'r--', lw=2)
            axes[idx].set_xlabel('Actual Price')
            axes[idx].set_ylabel('Predicted Price')
            axes[idx].set_title(f'{model_name}\nR² = {r2:.4f}')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(r"C:\Users\Admin\OneDrive\Documents\predictive housing price model\predictions_comparison.png", dpi=300, bbox_inches='tight')
        print("✓ Predictions comparison saved as 'predictions_comparison.png'")
        plt.close()
    
    def make_prediction(self, features_dict):
        """
        Make a prediction for a new house.
        
        Args:
            features_dict (dict): Dictionary with feature names and values
            
        Returns:
            dict: Predictions from all models
        """
        # Convert to DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Scale features if needed
        features_scaled = self.scaler.transform(features_df)
        
        predictions = {}
        
        for model_name, model in self.models.items():
            if model_name == 'Linear Regression':
                pred = model.predict(features_scaled)[0]
            else:
                pred = model.predict(features_df)[0]
            predictions[model_name] = pred
        
        return predictions
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        print("\n" + "="*80)
        print("HOUSING PRICE PREDICTION MODEL - FULL PIPELINE")
        print("="*80)
        
        self.load_data()
        self.explore_data()
        self.preprocess_data()
        self.train_models()
        self.evaluate_models()
        self.feature_importance()
        self.plot_predictions()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)


# Main execution
if __name__ == "__main__":
    # Initialize and run the predictor
    predictor = HousingPricePredictor()
    predictor.run_full_pipeline()
    
    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    
    sample_house = {
        'square_feet': 2500,
        'bedrooms': 4,
        'bathrooms': 2.5,
        'age_years': 10,
        'garage_spaces': 2,
        'lot_size': 8000,
        'condition': 4,
        'location_quality': 8
    }
    
    print(f"\nSample House Features:\n{pd.Series(sample_house)}")
    
    predictions = predictor.make_prediction(sample_house)
    print("\nPredicted Prices:")
    for model_name, price in predictions.items():
        print(f"  {model_name}: ${price:,.2f}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(predictor.results_df.to_string(index=False))
