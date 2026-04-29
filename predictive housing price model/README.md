# Predictive Housing Price Model

A machine learning project to predict house prices using Python, scikit-learn, and pandas. This project demonstrates the complete ML pipeline including data exploration, preprocessing, model training, evaluation, and prediction.

## 📋 Project Overview

This project builds and compares three regression models:
- **Linear Regression** - Baseline model for price prediction
- **Random Forest Regressor** - Ensemble method capturing non-linear relationships
- **Gradient Boosting Regressor** - Advanced ensemble method for optimal performance

## 🏗️ Project Structure

```
housing_price_model/
├── housing_price_model.py    # Main Python script with HousingPricePredictor class
├── requirements.txt           # Project dependencies
├── README.md                  # This file
├── data/                      # Directory for housing datasets (optional)
└── outputs/
    ├── correlation_matrix.png
    ├── price_analysis.png
    ├── feature_importance.png
    └── predictions_comparison.png
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Model
```bash
python housing_price_model.py
```

This will automatically:
- Generate a sample dataset (500 properties)
- Perform exploratory data analysis
- Preprocess the data
- Train three regression models
- Evaluate and compare models
- Generate visualizations
- Make example predictions

## 📊 Features

### Data Features (Predictors)
- **square_feet**: Total living area of the house
- **bedrooms**: Number of bedrooms
- **bathrooms**: Number of bathrooms
- **age_years**: Age of the house in years
- **garage_spaces**: Number of garage spaces
- **lot_size**: Total lot size in square feet
- **condition**: Overall condition rating (1-5 scale)
- **location_quality**: Location quality score (1-10 scale)

### Target Variable
- **price**: House price in dollars

## 🔍 Key Components

### HousingPricePredictor Class

#### Methods:

1. **load_data(filepath=None)**
   - Loads housing data from CSV or generates sample data
   - Displays basic statistics and info

2. **explore_data()**
   - Performs exploratory data analysis
   - Generates correlation matrix and price distributions
   - Creates visualizations

3. **preprocess_data()**
   - Handles missing values
   - Splits data into train/test sets (80/20)
   - Scales features using StandardScaler

4. **train_models()**
   - Trains Linear Regression
   - Trains Random Forest Regressor
   - Trains Gradient Boosting Regressor

5. **evaluate_models()**
   - Calculates R² score, RMSE, and MAE
   - Compares model performance
   - Returns results DataFrame

6. **feature_importance()**
   - Displays feature importance from tree-based models
   - Creates visualization comparing importances

7. **plot_predictions()**
   - Creates actual vs predicted plots
   - Shows model comparison visually

8. **make_prediction(features_dict)**
   - Makes predictions on new data
   - Returns predictions from all three models

9. **run_full_pipeline()**
   - Executes the complete ML pipeline

## 📈 Model Metrics

The project evaluates models using:

- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **RMSE**: Root Mean Squared Error (measures prediction error in dollars)
- **MAE**: Mean Absolute Error (average absolute prediction error)

Expected performance on test data:
- Linear Regression: R² ≈ 0.85
- Random Forest: R² ≈ 0.90
- Gradient Boosting: R² ≈ 0.92

## 💻 Usage Examples

### Basic Usage
```python
from housing_price_model import HousingPricePredictor

# Initialize predictor
predictor = HousingPricePredictor()

# Run full pipeline
predictor.run_full_pipeline()
```

### Make a Prediction
```python
# Define house features
new_house = {
    'square_feet': 3000,
    'bedrooms': 4,
    'bathrooms': 3,
    'age_years': 5,
    'garage_spaces': 2,
    'lot_size': 10000,
    'condition': 5,
    'location_quality': 9
}

# Get predictions from all models
predictions = predictor.make_prediction(new_house)

for model_name, price in predictions.items():
    print(f"{model_name}: ${price:,.2f}")
```

### Load Custom Data
```python
predictor = HousingPricePredictor()
predictor.load_data('your_housing_data.csv')
predictor.preprocess_data()
predictor.train_models()
predictor.evaluate_models()
```

### Using Specific Data
```python
# For CSV files with columns: square_feet, bedrooms, bathrooms, 
# age_years, garage_spaces, lot_size, condition, location_quality, price
predictor = HousingPricePredictor()
predictor.load_data('path/to/your/data.csv')
predictor.run_full_pipeline()
```

## 📊 Generated Visualizations

1. **correlation_matrix.png**
   - Heatmap of feature correlations
   - Shows relationships between all variables

2. **price_analysis.png**
   - Histogram of price distribution
   - Scatter plot of price vs square footage

3. **feature_importance.png**
   - Bar plots of feature importance
   - Compares Random Forest and Gradient Boosting importance

4. **predictions_comparison.png**
   - Actual vs predicted price scatter plots
   - Shows performance of all three models
   - Includes R² scores for each model

## 🎯 Performance Tips

### For Better Model Performance:
1. **Increase training data**: More examples improve accuracy
2. **Feature engineering**: Create new features from existing ones
3. **Hyperparameter tuning**: Use GridSearchCV to optimize parameters
4. **Data quality**: Clean data and handle outliers
5. **Cross-validation**: Use k-fold cross-validation for better estimates

### Hyperparameter Examples:
```python
# Random Forest tuning
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

# Gradient Boosting tuning
gb_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}
```

## 🔧 Advanced Usage

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

predictor = HousingPricePredictor()
predictor.load_data()
predictor.preprocess_data()

model = RandomForestRegressor(n_estimators=100)
cv_scores = cross_val_score(model, predictor.X_train, 
                            predictor.y_train, cv=5)
print(f"CV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f}")
```

### Hyperparameter Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, 20],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(GradientBoostingRegressor(),
                          param_grid, cv=5, n_jobs=-1)
grid_search.fit(predictor.X_train, predictor.y_train)
print(f"Best parameters: {grid_search.best_params_}")
```

## 📚 Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Install missing dependencies
```bash
pip install -r requirements.txt
```

### Issue: Memory Error with Large Datasets
**Solution**: Reduce dataset size or use feature selection
```python
# Use only top features
top_features = ['square_feet', 'bedrooms', 'bathrooms', 'condition']
X = df[top_features]
```

### Issue: Model Overfitting
**Solution**: Reduce model complexity
```python
# Reduce tree depth and complexity
rf_model = RandomForestRegressor(max_depth=10, min_samples_split=10)
```

## 📝 Project Workflow

```
1. Data Loading
   ↓
2. Exploratory Data Analysis (EDA)
   ↓
3. Data Preprocessing & Feature Scaling
   ↓
4. Train/Test Split
   ↓
5. Model Training (3 different models)
   ↓
6. Model Evaluation & Comparison
   ↓
7. Feature Importance Analysis
   ↓
8. Visualization & Results
   ↓
9. Making Predictions on New Data
```

## 🎓 Learning Outcomes

By working with this project, you'll learn:
- How to structure ML projects in Python
- Data exploration and preprocessing techniques
- Training and evaluating multiple regression models
- Model comparison and selection
- Feature importance analysis
- Creating professional visualizations
- Making predictions with trained models

## 📌 Notes

- The sample dataset is generated with realistic relationships between features and price
- Models achieve approximately 90%+ R² score on test data
- Random Forest and Gradient Boosting typically outperform Linear Regression
- All numeric features are scaled before training (except for tree-based models which don't require it)
- Train/test split is 80/20 with fixed random state for reproducibility

## 🚀 Next Steps

1. **Use real data**: Replace sample data with actual housing market data
2. **Feature engineering**: Create new features (e.g., price per sq ft)
3. **Hyperparameter tuning**: Use GridSearchCV for optimal parameters
4. **Advanced models**: Try XGBoost, LightGBM, or Neural Networks
5. **Deployment**: Create a Flask/FastAPI app for predictions

## 📄 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

Created as a comprehensive machine learning project example.

## 📞 Support

For issues or questions, refer to scikit-learn documentation:
- https://scikit-learn.org/stable/documentation.html
- https://pandas.pydata.org/docs/
