import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import re
import joblib
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ============================================
# 1. LOAD AND EXPLORE THE DATA
# ============================================

# Load the dataset
file_path = r"C:\Users\VIAMRSH P S\Desktop\Projects-20240722T093004Z-001\Projects\vehicle_price_prediction\Vehicle Price Prediction\dataset.csv"
df = pd.read_csv(file_path)

print("=" * 60)
print("VEHICLE PRICE PREDICTION - DATA EXPLORATION")
print("=" * 60)

print(f"\nDataset Shape: {df.shape}")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicated Rows: {df.duplicated().sum()}")

# ============================================
# 2. DATA CLEANING AND PREPROCESSING
# ============================================

print("\n" + "=" * 60)
print("DATA CLEANING & PREPROCESSING")
print("=" * 60)

# Create a copy for processing
data = df.copy()

# Clean price column (remove $ and commas, convert to numeric)
if data['price'].dtype == 'object':
    data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)

# Remove outliers in price (beyond 3 standard deviations)
price_mean = data['price'].mean()
price_std = data['price'].std()
data = data[data['price'].between(price_mean - 3*price_std, price_mean + 3*price_std)]

print(f"Shape after removing price outliers: {data.shape}")

# Clean mileage column
if data['mileage'].dtype == 'object':
    data['mileage'] = data['mileage'].replace('[,mi\s]', '', regex=True)
    data['mileage'] = pd.to_numeric(data['mileage'], errors='coerce')

# Extract engine size from engine column
def extract_engine_size(engine_str):
    if pd.isna(engine_str):
        return np.nan
    engine_str = str(engine_str).lower()
    if 'liter' in engine_str or 'l' in engine_str:
        try:
            # Extract number before 'l' or 'liter'
            match = re.search(r'(\d+\.?\d*)\s*[lL]', engine_str)
            if match:
                return float(match.group(1))
        except:
            pass
    return np.nan

data['engine_size'] = data['engine'].apply(extract_engine_size)

# Calculate vehicle age
current_year = 2024
data['vehicle_age'] = current_year - data['year']

# Handle missing values
print(f"\nMissing values before cleaning:\n{data.isnull().sum()}")

# Fill numeric missing values with median
numeric_cols = ['mileage', 'engine_size', 'cylinders', 'doors']
for col in numeric_cols:
    if col in data.columns:
        data[col].fillna(data[col].median(), inplace=True)

# Fill categorical missing values with mode
categorical_cols = ['fuel', 'transmission', 'drivetrain', 'body', 'exterior_color', 'interior_color']
for col in categorical_cols:
    if col in data.columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

print(f"\nMissing values after cleaning:\n{data.isnull().sum()}")

# ============================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Statistical summary
print(f"\nPrice Statistics:\n{data['price'].describe()}")

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Price distribution
axes[0,0].hist(data['price'], bins=50, color='skyblue', edgecolor='black')
axes[0,0].set_title('Price Distribution')
axes[0,0].set_xlabel('Price (USD)')
axes[0,0].set_ylabel('Frequency')

# Price vs Year
axes[0,1].scatter(data['year'], data['price'], alpha=0.5, color='coral')
axes[0,1].set_title('Price vs Year')
axes[0,1].set_xlabel('Year')
axes[0,1].set_ylabel('Price')

# Price vs Mileage
axes[0,2].scatter(data['mileage'], data['price'], alpha=0.5, color='green')
axes[0,2].set_title('Price vs Mileage')
axes[0,2].set_xlabel('Mileage')
axes[0,2].set_ylabel('Price')

# Price by Make (top 10)
top_makes = data['make'].value_counts().head(10).index
make_price = data[data['make'].isin(top_makes)].groupby('make')['price'].mean().sort_values(ascending=False)
axes[1,0].bar(range(len(make_price)), make_price.values, color='purple')
axes[1,0].set_title('Average Price by Make (Top 10)')
axes[1,0].set_xticks(range(len(make_price)))
axes[1,0].set_xticklabels(make_price.index, rotation=45, ha='right')
axes[1,0].set_ylabel('Average Price')

# Price by Body Type
body_price = data.groupby('body')['price'].mean().sort_values(ascending=False)
axes[1,1].bar(range(len(body_price)), body_price.values, color='orange')
axes[1,1].set_title('Average Price by Body Type')
axes[1,1].set_xticks(range(len(body_price)))
axes[1,1].set_xticklabels(body_price.index, rotation=45, ha='right')

# Price by Fuel Type
fuel_price = data.groupby('fuel')['price'].mean().sort_values(ascending=False)
axes[1,2].bar(range(len(fuel_price)), fuel_price.values, color='teal')
axes[1,2].set_title('Average Price by Fuel Type')
axes[1,2].set_xticks(range(len(fuel_price)))
axes[1,2].set_xticklabels(fuel_price.index, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('vehicle_eda.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nEDA visualization saved as 'vehicle_eda.png'")

# Correlation analysis
numeric_features = ['year', 'mileage', 'engine_size', 'cylinders', 'doors', 'vehicle_age', 'price']
corr_matrix = data[numeric_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nCorrelation matrix saved as 'correlation_matrix.png'")

# ============================================
# 4. FEATURE ENGINEERING
# ============================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Select features for modeling
features_to_use = ['make', 'model', 'year', 'mileage', 'fuel', 'transmission', 
                   'body', 'doors', 'drivetrain', 'engine_size', 'cylinders', 'vehicle_age']

# Create feature matrix
X = data[features_to_use].copy()
y = data['price'].copy()

print(f"Features selected: {features_to_use}")

# Encode categorical variables
categorical_features = ['make', 'model', 'fuel', 'transmission', 'body', 'drivetrain']
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"\nCategorical variables encoded: {categorical_features}")

# ============================================
# 5. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 6. MODEL TRAINING AND EVALUATION
# ============================================

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²:   {r2:.4f}")
    
    return model, y_pred, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=2),  # Limited to 2 cores
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
trained_models = {}

# Train and evaluate each model
for name, model in models.items():
    if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
        trained_model, predictions, metrics = evaluate_model(model, X_train_scaled, X_test_scaled, 
                                                            y_train, y_test, name)
    else:
        trained_model, predictions, metrics = evaluate_model(model, X_train, X_test, 
                                                            y_train, y_test, name)
    results[name] = metrics
    trained_models[name] = trained_model

# ============================================
# 7. MODEL COMPARISON
# ============================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results).T
print(results_df.sort_values('R2', ascending=False))

# Visualize model comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# R² Score comparison
axes[0].bar(results_df.index, results_df['R2'], color='steelblue')
axes[0].set_title('R² Score Comparison')
axes[0].set_ylabel('R² Score')
axes[0].tick_params(axis='x', rotation=45)

# RMSE comparison
axes[1].bar(results_df.index, results_df['RMSE'], color='coral')
axes[1].set_title('RMSE Comparison')
axes[1].set_ylabel('RMSE (USD)')
axes[1].tick_params(axis='x', rotation=45)

# MAE comparison
axes[2].bar(results_df.index, results_df['MAE'], color='lightgreen')
axes[2].set_title('MAE Comparison')
axes[2].set_ylabel('MAE (USD)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nModel comparison saved as 'model_comparison.png'")

# ============================================
# 8. BEST MODEL ANALYSIS (Random Forest usually performs best)
# ============================================

print("\n" + "=" * 60)
print("BEST MODEL ANALYSIS")
print("=" * 60)

# Select best model based on R²
best_model_name = results_df['R2'].idxmax()
best_model = trained_models[best_model_name]

print(f"Best Model: {best_model_name}")

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'][:10][::-1], feature_importance['importance'][:10][::-1])
    plt.title(f'Top 10 Feature Importances - {best_model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nFeature importance saved as 'feature_importance.png'")

# Predictions vs Actual
y_pred_best = best_model.predict(X_test if best_model_name not in ['Linear Regression', 'Ridge Regression', 'Lasso Regression'] else X_test_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Actual vs Predicted Prices - {best_model_name}')
plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPredictions vs Actual plot saved as 'predictions_vs_actual.png'")

# ============================================
# 9. HYPERPARAMETER TUNING (Memory-Optimized)
# ============================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

# Use RandomizedSearchCV instead of GridSearchCV to reduce memory usage
print("Tuning Random Forest with RandomizedSearchCV...")

param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

try:
    # Use RandomizedSearchCV with limited iterations and single job to prevent memory issues
    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),  # Single job to save memory
        param_distributions=param_dist,
        n_iter=10,  # Only try 10 random combinations instead of all 81
        cv=2,  # Reduced from 3 to 2 folds
        scoring='neg_mean_absolute_error',
        n_jobs=1,  # Single job to prevent memory issues
        verbose=1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best CV Score (MAE): {-random_search.best_score_:,.2f}")

    # Evaluate tuned model
    tuned_rf = random_search.best_estimator_
    y_pred_tuned = tuned_rf.predict(X_test)

    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    r2_tuned = r2_score(y_test, y_pred_tuned)

    print(f"\nTuned Random Forest Results:")
    print(f"  MAE:  ${mae_tuned:,.2f}")
    print(f"  RMSE: ${rmse_tuned:,.2f}")
    print(f"  R²:   {r2_tuned:.4f}")
    
except Exception as e:
    print(f"\nHyperparameter tuning failed due to: {str(e)}")
    print("Using the best model from initial training instead.")
    tuned_rf = best_model

# ============================================
# 10. PREDICTION FUNCTION
# ============================================

print("\n" + "=" * 60)
print("PREDICTION FUNCTION")
print("=" * 60)

def predict_vehicle_price(make, model, year, mileage, fuel, transmission, 
                         body, doors, drivetrain, engine_size, cylinders):
    """
    Predict vehicle price using the trained model
    """
    # Create input dataframe
    input_data = pd.DataFrame([{
        'make': make,
        'model': model,
        'year': year,
        'mileage': mileage,
        'fuel': fuel,
        'transmission': transmission,
        'body': body,
        'doors': doors,
        'drivetrain': drivetrain,
        'engine_size': engine_size,
        'cylinders': cylinders,
        'vehicle_age': current_year - year
    }])
    
    # Encode categorical variables
    for col in categorical_features:
        if col in label_encoders:
            try:
                input_data[col] = label_encoders[col].transform([str(input_data[col].values[0])])
            except ValueError:
                # Handle unseen categories
                input_data[col] = 0
    
    # Make prediction
    prediction = tuned_rf.predict(input_data)[0]
    return prediction

# Example prediction
example_vehicle = {
    'make': 'Toyota',
    'model': 'Camry',
    'year': 2020,
    'mileage': 30000,
    'fuel': 'Gasoline',
    'transmission': 'Automatic',
    'body': 'Sedan',
    'doors': 4,
    'drivetrain': 'Front-wheel Drive',
    'engine_size': 2.5,
    'cylinders': 4
}

predicted_price = predict_vehicle_price(**example_vehicle)
print(f"\nExample Prediction:")
print(f"Vehicle: {example_vehicle['year']} {example_vehicle['make']} {example_vehicle['model']}")
print(f"Predicted Price: ${predicted_price:,.2f}")

# ============================================
# 11. SAVE MODEL
# ============================================

# Save the model and preprocessing objects
joblib.dump(tuned_rf, 'vehicle_price_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)
print("Files saved:")
print("  - vehicle_price_model.pkl (Trained model)")
print("  - label_encoders.pkl (Label encoders for categorical variables)")
print("  - scaler.pkl (Feature scaler)")
print("  - vehicle_eda.png (EDA visualizations)")
print("  - correlation_matrix.png (Correlation matrix)")
print("  - model_comparison.png (Model comparison)")
print("  - feature_importance.png (Feature importance)")
print("  - predictions_vs_actual.png (Prediction accuracy)")

print("\n" + "=" * 60)
print("VEHICLE PRICE PREDICTION SYSTEM COMPLETE")
print("=" * 60)
