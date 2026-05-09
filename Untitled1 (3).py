# Generated from: Untitled1 (3).ipynb
# Converted at: 2026-05-09T06:44:01.331Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ============================================================
# Smart Retail Demand Forecasting & Customer Analytics System
# Author: Retail Analytics Engine
# Compatible with: Google Colab
# ============================================================

# Step 1: Install required libraries (Colab friendly)
!pip install xgboost scikit-learn pandas numpy matplotlib seaborn openpyxl -q

# Step 2: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully")

# ============================================================
# Step 3: Generate Synthetic Retail Data (50,000+ rows)
# ============================================================
np.random.seed(42)

# Date range
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')

# Product categories
categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Sports', 'Beauty', 'Toys', 'Groceries']
products = {
    'Electronics': ['Laptop', 'Smartphone', 'Headphones', 'Smartwatch'],
    'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers'],
    'Home & Kitchen': ['Blender', 'Microwave', 'Vacuum', 'Toaster'],
    'Sports': ['Yoga Mat', 'Dumbbells', 'Treadmill', 'Basketball'],
    'Beauty': ['Shampoo', 'Perfume', 'Lipstick', 'Face Cream'],
    'Toys': ['Lego', 'Doll', 'Action Figure', 'Board Game'],
    'Groceries': ['Rice', 'Milk', 'Bread', 'Eggs']
}

# Generate transaction data
transactions = []
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)

for _ in range(50000):  # 50,000 transactions
    date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))
    category = np.random.choice(categories, p=[0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1])
    product = np.random.choice(products[category])
    quantity = np.random.randint(1, 5)
    unit_price = np.random.uniform(10, 500)
    if category == 'Electronics':
        unit_price = np.random.uniform(200, 1500)
    elif category == 'Groceries':
        unit_price = np.random.uniform(2, 20)

    total_price = quantity * unit_price

    # Customer info
    customer_id = np.random.randint(1, 2000)
    age = np.random.randint(18, 70)
    gender = np.random.choice(['Male', 'Female'])
    city = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])
    loyalty_tier = np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], p=[0.5, 0.3, 0.15, 0.05])

    transactions.append({
        'TransactionID': f"TXN_{_}",
        'Date': date,
        'CustomerID': customer_id,
        'Category': category,
        'Product': product,
        'Quantity': quantity,
        'UnitPrice': unit_price,
        'TotalPrice': total_price,
        'Age': age,
        'Gender': gender,
        'City': city,
        'LoyaltyTier': loyalty_tier
    })

df = pd.DataFrame(transactions)
print(f"✅ Synthetic data generated: {len(df):,} transactions")
print(df.head())

# ============================================================
# Step 4: Exploratory Data Analysis (EDA)
# ============================================================
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Basic info
print("\nDataset Info:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nBasic Statistics:\n", df.describe())

# Sales trends over time
daily_sales = df.groupby('Date')['TotalPrice'].sum().reset_index()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
daily_sales.set_index('Date')['TotalPrice'].plot()
plt.title('Daily Sales Trend')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')

# Sales by category
plt.subplot(1, 3, 2)
category_sales = df.groupby('Category')['TotalPrice'].sum().sort_values()
category_sales.plot(kind='barh')
plt.title('Total Sales by Category')
plt.xlabel('Revenue ($)')

# Sales by loyalty tier
plt.subplot(1, 3, 3)
loyalty_sales = df.groupby('LoyaltyTier')['TotalPrice'].sum().sort_values()
loyalty_sales.plot(kind='bar', color=['gold', 'silver', 'brown', 'gray'])
plt.title('Sales by Loyalty Tier')
plt.ylabel('Revenue ($)')
plt.tight_layout()
plt.show()

# ============================================================
# Step 5: Feature Engineering for Demand Forecasting
# ============================================================
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Aggregate daily sales per product
daily_product_sales = df.groupby(['Date', 'Product', 'Category']).agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum'
}).reset_index()

# Add time-based features
daily_product_sales['DayOfWeek'] = daily_product_sales['Date'].dt.dayofweek
daily_product_sales['Month'] = daily_product_sales['Date'].dt.month
daily_product_sales['Quarter'] = daily_product_sales['Date'].dt.quarter
daily_product_sales['IsWeekend'] = daily_product_sales['DayOfWeek'].isin([5, 6]).astype(int)
daily_product_sales['DayOfYear'] = daily_product_sales['Date'].dt.dayofyear

# Lag features (demand from previous days)
for product in daily_product_sales['Product'].unique()[:5]:  # Sample for demonstration
    prod_mask = daily_product_sales['Product'] == product
    daily_product_sales.loc[prod_mask, 'Lag1'] = daily_product_sales.loc[prod_mask, 'Quantity'].shift(1)
    daily_product_sales.loc[prod_mask, 'Lag7'] = daily_product_sales.loc[prod_mask, 'Quantity'].shift(7)

# Rolling averages
daily_product_sales['RollingMean7'] = daily_product_sales.groupby('Product')['Quantity'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

print("✅ Feature engineering completed")
print(f"Final feature set shape: {daily_product_sales.shape}")

# ============================================================
# Step 6: Demand Forecasting Model (ML)
# ============================================================
print("\n" + "="*60)
print("DEMAND FORECASTING MODELS")
print("="*60)

# Prepare data for modeling
model_data = daily_product_sales.dropna(subset=['Lag1', 'Lag7']).copy()
features = ['DayOfWeek', 'Month', 'Quarter', 'IsWeekend', 'Lag1', 'Lag7', 'RollingMean7']
target = 'Quantity'

# Encode categorical variables
le = LabelEncoder()
model_data['Product_encoded'] = le.fit_transform(model_data['Product'])
features.append('Product_encoded')

X = model_data[features]
y = model_data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"\n{name}:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R² Score: {r2:.4f}")

# Best model
best_model_name = max(results, key=lambda x: results[x]['R2'])
print(f"\n🏆 Best Model: {best_model_name}")

# ============================================================
# Step 7: Customer Segmentation (K-Means Clustering)
# ============================================================
print("\n" + "="*60)
print("CUSTOMER SEGMENTATION (RFM + CLUSTERING)")
print("="*60)

# RFM Analysis
snapshot_date = df['Date'].max()
rfm = df.groupby('CustomerID').agg({
    'Date': lambda x: (snapshot_date - x.max()).days,  # Recency
    'TransactionID': 'count',  # Frequency
    'TotalPrice': 'sum'  # Monetary
}).rename(columns={
    'Date': 'Recency',
    'TransactionID': 'Frequency',
    'TotalPrice': 'Monetary'
})

# Add additional features
customer_demographics = df.groupby('CustomerID').agg({
    'Age': 'first',
    'Gender': 'first',
    'City': 'first',
    'LoyaltyTier': 'first'
})

rfm = rfm.join(customer_demographics)
rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']

# Handle infinities
rfm.replace([np.inf, -np.inf], 0, inplace=True)
rfm.fillna(0, inplace=True)

# Scale features for clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary', 'AvgOrderValue']])

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

# Segment interpretation
segment_profiles = rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'AvgOrderValue': 'mean'
}).round(2)

print("\nSegment Profiles:")
print(segment_profiles)

# Visualize clusters
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm['Segment'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('Customer Segments (PCA Projection)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# ============================================================
# Step 8: Inventory Planning Insights
# ============================================================
print("\n" + "="*60)
print("INVENTORY PLANNING INSIGHTS")
print("="*60)

# Seasonal demand patterns
seasonal_demand = df.groupby([df['Date'].dt.month, 'Category'])['Quantity'].sum().unstack()
plt.figure(figsize=(12, 6))
sns.heatmap(seasonal_demand, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('Seasonal Demand Heatmap: Monthly Quantity by Category')
plt.xlabel('Category')
plt.ylabel('Month')
plt.show()

# Top products by demand volatility
product_volatility = df.groupby('Product')['Quantity'].agg(['mean', 'std'])
product_volatility['CV'] = product_volatility['std'] / product_volatility['mean']  # Coefficient of Variation
top_volatile = product_volatility.nlargest(10, 'CV')
print("\nTop 10 Most Volatile Products (high inventory risk):")
print(top_volatile[['mean', 'std', 'CV']])

# Safety stock recommendation (simplified)
product_volatility['SafetyStock'] = product_volatility['std'] * 1.65  # 95% service level
print("\nRecommended Safety Stock Levels (95% service level):")
print(product_volatility[['mean', 'SafetyStock']].head(10))

# ============================================================
# Step 9: Export for Power BI
# ============================================================
print("\n" + "="*60)
print("EXPORTING DATA FOR POWER BI DASHBOARDS")
print("="*60)

# Create aggregated views for Power BI
powerbi_export = {
    'fact_transactions': df,
    'dim_products': df[['Product', 'Category']].drop_duplicates().reset_index(drop=True),
    'dim_customers': rfm[['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'Segment', 'LoyaltyTier']].reset_index(),
    'daily_sales': daily_sales,
    'forecast_results': pd.DataFrame({
        'Actual': y_test.values[:100],
        'Predicted': models[best_model_name].predict(X_test)[:100]
    }),
    'segment_profiles': segment_profiles.reset_index()
}

# Save to Excel (Power BI compatible)
with pd.ExcelWriter('retail_analytics_powerbi.xlsx', engine='openpyxl') as writer:
    for sheet_name, data in powerbi_export.items():
        if isinstance(data, pd.DataFrame) and len(data) > 0:
            data.to_excel(writer, sheet_name=sheet_name[:31], index=False)

print("✅ Exported 'retail_analytics_powerbi.xlsx' with 6 sheets")

# Also save CSV for backup
df.to_csv('retail_transactions.csv', index=False)
print("✅ Saved 'retail_transactions.csv'")

# ============================================================
# Step 10: Summary Report
# ============================================================
print("\n" + "="*60)
print("SYSTEM SUMMARY REPORT")
print("="*60)

print(f"""
📊 DATA OVERVIEW:
   - Total Transactions: {len(df):,}
   - Unique Customers: {df['CustomerID'].nunique():,}
   - Products: {df['Product'].nunique()}
   - Categories: {df['Category'].nunique()}
   - Date Range: {df['Date'].min()} to {df['Date'].max()}

💰 SALES METRICS:
   - Total Revenue: ${df['TotalPrice'].sum():,.2f}
   - Average Order Value: ${df['TotalPrice'].mean():.2f}
   - Total Units Sold: {df['Quantity'].sum():,}

🤖 MODEL PERFORMANCE:
   - Best Model: {best_model_name}
   - R² Score: {results[best_model_name]['R2']:.4f}
   - MAE: {results[best_model_name]['MAE']:.2f} units

👥 CUSTOMER SEGMENTS:
   - Segments Created: {len(segment_profiles)}
   - Segment Distribution:
{rfm['Segment'].value_counts().sort_index().to_string()}

📁 EXPORTED FILES:
   1. retail_analytics_powerbi.xlsx → Import directly into Power BI
   2. retail_transactions.csv → Raw transaction data

✅ System ready for business intelligence and inventory planning.
""")

print("\n🚀 Analytics pipeline completed successfully!")

pip install openpyxl

# ============================================================
# COMPLETE FIXED VERSION - RETAIL ANALYTICS SYSTEM
# Run this entire code in Google Colab
# ============================================================

# Step 1: Install required libraries
!pip install prophet xgboost lightgbm optuna pmdarima -q

# Step 2: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time Series & Forecasting
from prophet import Prophet
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Machine Learning
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Optimization
import optuna
from itertools import combinations

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

print("✅ Libraries loaded successfully")

# ============================================================
# DATA GENERATION (FIXED)
# ============================================================

def generate_enhanced_retail_data(n_transactions=50000):
    """Generate realistic retail data"""

    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days

    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Sports', 'Beauty', 'Groceries']

    products_by_category = {
        'Electronics': ['Laptop', 'Smartphone', 'Headphones', 'Tablet'],
        'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers'],
        'Home & Kitchen': ['Blender', 'Microwave', 'Vacuum', 'Toaster'],
        'Sports': ['Yoga Mat', 'Dumbbells', 'Treadmill', 'Basketball'],
        'Beauty': ['Shampoo', 'Perfume', 'Lipstick', 'Face Cream'],
        'Groceries': ['Rice', 'Milk', 'Bread', 'Eggs']
    }

    transactions = []

    for i in range(n_transactions):
        # Random date
        random_days = np.random.randint(0, date_range)
        date = start_date + timedelta(days=random_days)

        # Category with seasonal bias
        month = date.month
        if month in [11, 12]:  # Holiday season
            category_weights = [0.3, 0.2, 0.15, 0.1, 0.15, 0.1]
        else:
            category_weights = [0.2, 0.25, 0.15, 0.1, 0.15, 0.15]

        category = np.random.choice(categories, p=category_weights)
        product = np.random.choice(products_by_category[category])

        # Price based on category
        if category == 'Electronics':
            price = np.random.uniform(200, 1500)
        elif category == 'Groceries':
            price = np.random.uniform(2, 20)
        else:
            price = np.random.uniform(20, 200)

        # Quantity
        quantity = np.random.randint(1, 4)

        # Holiday multiplier
        is_holiday = 1.5 if (month == 11 and date.day >= 20) or (month == 12 and date.day <= 26) else 1.0

        # Discount
        discount = np.random.uniform(0, 0.2) if np.random.random() < 0.3 else 0

        final_price = price * quantity * (1 - discount) * is_holiday

        # Customer
        customer_id = np.random.randint(1, 2000)

        transactions.append({
            'TransactionID': f'TXN_{i:06d}',
            'Date': date,
            'CustomerID': customer_id,
            'Category': category,
            'Product': product,
            'Quantity': quantity,
            'UnitPrice': price,
            'Discount': discount,
            'TotalPrice': final_price,
            'IsHoliday': is_holiday > 1
        })

    return pd.DataFrame(transactions)

print("Generating retail data...")
df = generate_enhanced_retail_data(50000)
print(f"✅ Generated {len(df):,} transactions")
print(df.head())

# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Basic statistics
print(f"\nTotal Revenue: ${df['TotalPrice'].sum():,.2f}")
print(f"Average Order Value: ${df['TotalPrice'].mean():.2f}")
print(f"Total Transactions: {len(df):,}")
print(f"Unique Customers: {df['CustomerID'].nunique():,}")
print(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# Sales by category
category_sales = df.groupby('Category')['TotalPrice'].sum().sort_values(ascending=False)
print("\nSales by Category:")
print(category_sales)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Daily sales trend
daily_sales = df.groupby('Date')['TotalPrice'].sum()
axes[0,0].plot(daily_sales.index, daily_sales.values, alpha=0.7, linewidth=1)
axes[0,0].set_title('Daily Sales Trend', fontweight='bold')
axes[0,0].set_xlabel('Date')
axes[0,0].set_ylabel('Revenue ($)')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Sales by category
category_sales.plot(kind='barh', ax=axes[0,1], color='coral')
axes[0,1].set_title('Total Revenue by Category', fontweight='bold')
axes[0,1].set_xlabel('Revenue ($)')

# 3. Hourly/Day pattern
df['DayOfWeek'] = df['Date'].dt.dayofweek
weekly_sales = df.groupby('DayOfWeek')['TotalPrice'].mean()
axes[1,0].bar(weekly_sales.index, weekly_sales.values, color='skyblue')
axes[1,0].set_title('Average Sales by Day of Week', fontweight='bold')
axes[1,0].set_xticks(range(7))
axes[1,0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
axes[1,0].set_ylabel('Avg Revenue ($)')

# 4. Top products
top_products = df.groupby('Product')['TotalPrice'].sum().nlargest(10)
axes[1,1].barh(range(len(top_products)), top_products.values, color='lightgreen')
axes[1,1].set_yticks(range(len(top_products)))
axes[1,1].set_yticklabels(top_products.index)
axes[1,1].set_title('Top 10 Products by Revenue', fontweight='bold')
axes[1,1].set_xlabel('Revenue ($)')

plt.tight_layout()
plt.show()

# ============================================================
# TIME SERIES FORECASTING WITH PROPHET
# ============================================================

print("\n" + "="*60)
print("TIME SERIES FORECASTING")
print("="*60)

# Prepare data for Prophet
prophet_df = df.groupby('Date')['TotalPrice'].sum().reset_index()
prophet_df.columns = ['ds', 'y']

# Train Prophet model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(prophet_df)

# Make future predictions
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
plt.title('Revenue Forecast - Next 90 Days', fontweight='bold')
plt.show()

fig2 = model.plot_components(forecast)
plt.show()

print(f"Forecast for next 90 days generated")
print(f"Predicted revenue for next 30 days: ${forecast['yhat'].iloc[-90:-60].sum():,.2f}")

# ============================================================
# FEATURE ENGINEERING FOR ML
# ============================================================

print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Create features
daily_features = df.groupby('Date').agg({
    'TotalPrice': 'sum',
    'Quantity': 'sum',
    'CustomerID': 'nunique',
    'TransactionID': 'count'
}).rename(columns={
    'TotalPrice': 'revenue',
    'Quantity': 'units',
    'CustomerID': 'unique_customers',
    'TransactionID': 'transactions'
})

# Time features
daily_features['dayofweek'] = daily_features.index.dayofweek
daily_features['month'] = daily_features.index.month
daily_features['quarter'] = daily_features.index.quarter
daily_features['dayofyear'] = daily_features.index.dayofyear
daily_features['is_weekend'] = (daily_features['dayofweek'] >= 5).astype(int)

# Lag features
for lag in [1, 2, 3, 7, 14]:
    daily_features[f'revenue_lag_{lag}'] = daily_features['revenue'].shift(lag)

# Rolling statistics
for window in [7, 14]:
    daily_features[f'revenue_rolling_mean_{window}'] = daily_features['revenue'].rolling(window).mean()
    daily_features[f'revenue_rolling_std_{window}'] = daily_features['revenue'].rolling(window).std()

# Drop NaN values
daily_features = daily_features.dropna()

print(f"Features created: {daily_features.shape}")
print(f"Feature columns: {list(daily_features.columns)}")

# ============================================================
# XGBOOST MODEL WITH OPTUNA
# ============================================================

print("\n" + "="*60)
print("XGBOOST WITH HYPERPARAMETER TUNING")
print("="*60)

# Prepare data
target = 'revenue'
feature_cols = [col for col in daily_features.columns if col != target]

X = daily_features[feature_cols]
y = daily_features[target]

# Split data
split_idx = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Define objective function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))

# Run optimization
print("Optimizing XGBoost...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, show_progress_bar=True)

print(f"\nBest parameters: {study.best_params}")
print(f"Best RMSE: {study.best_value:.2f}")

# Train final model with best parameters
best_xgb = xgb.XGBRegressor(**study.best_params, random_state=42)
best_xgb.fit(X_train, y_train)

# Evaluate
y_pred_xgb = best_xgb.predict(X_val)
xgb_mae = mean_absolute_error(y_val, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
xgb_mape = mean_absolute_percentage_error(y_val, y_pred_xgb) * 100

print(f"\nXGBoost Performance:")
print(f"MAE: ${xgb_mae:.2f}")
print(f"RMSE: ${xgb_rmse:.2f}")
print(f"MAPE: {xgb_mape:.1f}%")

# ============================================================
# LSTM DEEP LEARNING MODEL
# ============================================================

print("\n" + "="*60)
print("LSTM DEEP LEARNING MODEL")
print("="*60)

# Prepare sequences for LSTM
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Use revenue data
revenue_series = daily_features['revenue'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
revenue_scaled = scaler.fit_transform(revenue_series)

# Create sequences
seq_length = 30
X_lstm, y_lstm = create_sequences(revenue_scaled, seq_length)

# Split
split_lstm = int(len(X_lstm) * 0.8)
X_train_lstm, X_val_lstm = X_lstm[:split_lstm], X_lstm[split_lstm:]
y_train_lstm, y_val_lstm = y_lstm[:split_lstm], y_lstm[split_lstm:]

print(f"LSTM training samples: {len(X_train_lstm)}")
print(f"LSTM validation samples: {len(X_val_lstm)}")

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(25, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Callbacks
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

# Train
print("Training LSTM model...")
history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    validation_data=(X_val_lstm, y_val_lstm),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

# Evaluate
y_pred_lstm_scaled = lstm_model.predict(X_val_lstm)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled)
y_val_lstm_actual = scaler.inverse_transform(y_val_lstm.reshape(-1, 1))

lstm_mae = mean_absolute_error(y_val_lstm_actual, y_pred_lstm)
lstm_rmse = np.sqrt(mean_squared_error(y_val_lstm_actual, y_pred_lstm))
lstm_mape = mean_absolute_percentage_error(y_val_lstm_actual, y_pred_lstm) * 100

print(f"\nLSTM Performance:")
print(f"MAE: ${lstm_mae:.2f}")
print(f"RMSE: ${lstm_rmse:.2f}")
print(f"MAPE: {lstm_mape:.1f}%")

# ============================================================
# CUSTOMER SEGMENTATION (RFM Analysis)
# ============================================================

print("\n" + "="*60)
print("CUSTOMER SEGMENTATION (RFM)")
print("="*60)

# Calculate RFM metrics
snapshot_date = df['Date'].max()

rfm = df.groupby('CustomerID').agg({
    'Date': lambda x: (snapshot_date - x.max()).days,
    'TransactionID': 'count',
    'TotalPrice': 'sum'
}).rename(columns={
    'Date': 'Recency',
    'TransactionID': 'Frequency',
    'TotalPrice': 'Monetary'
})

# Add average order value
rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']

# Create RFM scores
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=['4', '3', '2', '1'])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=['1', '2', '3', '4'])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=['1', '2', '3', '4'])

# Combine scores
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# Segment customers
def segment_customers(row):
    if row['RFM_Score'] in ['444', '443', '434', '344']:
        return 'Champions'
    elif row['RFM_Score'] in ['442', '441', '432', '431']:
        return 'Loyal Customers'
    elif row['RFM_Score'] in ['333', '334', '343', '433']:
        return 'Potential Loyalists'
    elif row['RFM_Score'] in ['411', '412', '421', '422']:
        return 'New Customers'
    elif row['RFM_Score'] in ['311', '312', '321', '322']:
        return 'Promising'
    elif row['RFM_Score'] in ['211', '212', '221', '222']:
        return 'Need Attention'
    else:
        return 'At Risk'

rfm['Segment'] = rfm.apply(segment_customers, axis=1)

print("\nCustomer Segment Distribution:")
segment_counts = rfm['Segment'].value_counts()
for segment, count in segment_counts.items():
    print(f"  {segment}: {count} customers ({count/len(rfm)*100:.1f}%)")

print(f"\nTotal Customers Analyzed: {len(rfm):,}")
print(f"Average Customer Value: ${rfm['Monetary'].mean():.2f}")
print(f"Total Customer Value: ${rfm['Monetary'].sum():,.2f}")

# ============================================================
# MARKET BASKET ANALYSIS
# ============================================================

print("\n" + "="*60)
print("MARKET BASKET ANALYSIS")
print("="*60)

# Sample for basket analysis
baskets = df.groupby('TransactionID')['Product'].agg(list).reset_index()

# Find frequent product pairs
from collections import Counter

product_pairs = []
for basket in baskets['Product'].head(2000):  # Sample 2000 transactions
    for i in range(len(basket)):
        for j in range(i+1, len(basket)):
            pair = tuple(sorted([basket[i], basket[j]]))
            product_pairs.append(pair)

# Count pairs
pair_counts = Counter(product_pairs)
total_baskets = 2000

# Create pairs dataframe
pairs_df = pd.DataFrame([(p1, p2, count/total_baskets) for (p1, p2), count in pair_counts.items()],
                       columns=['Product_A', 'Product_B', 'Support'])
pairs_df = pairs_df[pairs_df['Support'] >= 0.03].sort_values('Support', ascending=False)

print(f"\nTop Product Pairs Frequently Bought Together:")
if len(pairs_df) > 0:
    for i, row in pairs_df.head(10).iterrows():
        print(f"  {row['Product_A']} & {row['Product_B']}: {row['Support']*100:.1f}% of transactions")
else:
    print("  No strong product pairs found")

# ============================================================
# INVENTORY OPTIMIZATION
# ============================================================

print("\n" + "="*60)
print("INVENTORY OPTIMIZATION")
print("="*60)

# Calculate product demand metrics
product_demand = df.groupby('Product').agg({
    'Quantity': ['mean', 'std', 'sum'],
    'TransactionID': 'count'
}).reset_index()

product_demand.columns = ['Product', 'Avg_Daily_Demand', 'Std_Daily_Demand', 'Total_Demand', 'Frequency']

# Calculate safety stock (95% service level)
from scipy.stats import norm
z_score = norm.ppf(0.95)  # 1.645 for 95% service level
lead_time_days = 7  # Assume 7 days lead time

product_demand['Safety_Stock'] = z_score * product_demand['Std_Daily_Demand'] * np.sqrt(lead_time_days)
product_demand['Reorder_Point'] = (product_demand['Avg_Daily_Demand'] * lead_time_days) + product_demand['Safety_Stock']

# Top products by demand
print("\nTop 10 Products - Safety Stock Recommendations:")
top_demand_products = product_demand.nlargest(10, 'Total_Demand')
for _, row in top_demand_products.iterrows():
    print(f"  {row['Product']}:")
    print(f"    Daily Demand: {row['Avg_Daily_Demand']:.1f} units")
    print(f"    Safety Stock: {row['Safety_Stock']:.0f} units")
    print(f"    Reorder Point: {row['Reorder_Point']:.0f} units")

# ============================================================
# COMPREHENSIVE DASHBOARD
# ============================================================

print("\n" + "="*60)
print("FINAL DASHBOARD - ALL VISUALIZATIONS")
print("="*60)

# Create comprehensive dashboard
fig = plt.figure(figsize=(20, 12))

# 1. Revenue vs Forecast
ax1 = plt.subplot(2, 3, 1)
ax1.plot(daily_sales.index[-90:], daily_sales.values[-90:], label='Actual', alpha=0.7)
ax1.plot(forecast['ds'].iloc[-90:], forecast['yhat'].iloc[-90:],
         label='Forecast', linestyle='--', linewidth=2)
ax1.set_title('Revenue: Actual vs Forecast', fontweight='bold')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# 2. Customer Segments Pie
ax2 = plt.subplot(2, 3, 2)
segment_revenue = rfm.groupby('Segment')['Monetary'].sum()
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c', '#e67e22']
ax2.pie(segment_revenue.values, labels=segment_revenue.index, autopct='%1.1f%%', colors=colors[:len(segment_revenue)])
ax2.set_title('Revenue by Customer Segment', fontweight='bold')

# 3. Model Performance Comparison
ax3 = plt.subplot(2, 3, 3)
models = ['XGBoost', 'LSTM']
mape_values = [xgb_mape, lstm_mape]
bars = ax3.bar(models, mape_values, color=['#e74c3c', '#3498db'])
ax3.set_ylabel('MAPE (%)')
ax3.set_title('Model Performance (Lower is Better)', fontweight='bold')
for bar, value in zip(bars, mape_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{value:.1f}%', ha='center', fontweight='bold')

# 4. Monthly Sales Heatmap
ax4 = plt.subplot(2, 3, 4)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
monthly_sales = df.pivot_table(values='TotalPrice', index='Month', columns='Year', aggfunc='sum')
im = ax4.imshow(monthly_sales.values, cmap='YlOrRd', aspect='auto')
ax4.set_xticks(range(len(monthly_sales.columns)))
ax4.set_xticklabels(monthly_sales.columns)
ax4.set_yticks(range(len(monthly_sales.index)))
ax4.set_yticklabels(monthly_sales.index)
ax4.set_title('Monthly Sales Heatmap', fontweight='bold')
ax4.set_xlabel('Year')
ax4.set_ylabel('Month')
plt.colorbar(im, ax=ax4, label='Revenue ($)')

# 5. Top Categories
ax5 = plt.subplot(2, 3, 5)
category_sales_sorted = category_sales.sort_values()
ax5.barh(category_sales_sorted.index, category_sales_sorted.values, color='coral')
ax5.set_title('Revenue by Category', fontweight='bold')
ax5.set_xlabel('Revenue ($)')

# 6. Weekly Pattern
ax6 = plt.subplot(2, 3, 6)
weekly_avg = df.groupby('DayOfWeek')['TotalPrice'].mean()
ax6.plot(weekly_avg.index, weekly_avg.values, marker='o', linewidth=2, markersize=8, color='green')
ax6.set_xticks(range(7))
ax6.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax6.set_title('Average Revenue by Day', fontweight='bold')
ax6.set_ylabel('Avg Revenue ($)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# EXPORT FOR POWER BI
# ============================================================

print("\n" + "="*60)
print("EXPORTING DATA FOR POWER BI")
print("="*60)

# Create Excel file with multiple sheets
with pd.ExcelWriter('retail_analytics_complete.xlsx', engine='openpyxl') as writer:
    # Raw data sample
    df.head(10000).to_excel(writer, sheet_name='Transactions', index=False)

    # RFM Analysis
    rfm.to_excel(writer, sheet_name='Customer_RFM')

    # Daily metrics
    daily_features.to_excel(writer, sheet_name='Daily_Metrics')

    # Product performance
    product_performance = df.groupby(['Category', 'Product']).agg({
        'TotalPrice': 'sum',
        'Quantity': 'sum',
        'TransactionID': 'count'
    }).rename(columns={
        'TotalPrice': 'Total_Revenue',
        'Quantity': 'Total_Units',
        'TransactionID': 'Transaction_Count'
    })
    product_performance.to_excel(writer, sheet_name='Product_Performance')

    # Forecast
    forecast_export = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(90)
    forecast_export.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound']
    forecast_export.to_excel(writer, sheet_name='Revenue_Forecast', index=False)

    # Inventory recommendations
    product_demand.to_excel(writer, sheet_name='Inventory_Optimization', index=False)

print("✅ Exported 'retail_analytics_complete.xlsx'")
print("✅ Excel file ready for Power BI import")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"""
╔════════════════════════════════════════════════════════════╗
║           RETAIL ANALYTICS SYSTEM - COMPLETE              ║
╚════════════════════════════════════════════════════════════╝

📊 DATA SUMMARY:
   • Total Transactions: {len(df):,}
   • Total Revenue: ${df['TotalPrice'].sum():,.2f}
   • Unique Customers: {df['CustomerID'].nunique():,}
   • Products: {df['Product'].nunique()}
   • Categories: {df['Category'].nunique()}

🎯 MODEL PERFORMANCE:
   • XGBoost MAPE: {xgb_mape:.1f}%
   • LSTM MAPE: {lstm_mape:.1f}%
   • Best Model: {'XGBoost' if xgb_mape < lstm_mape else 'LSTM'}

👥 CUSTOMER INSIGHTS:
   • Customer Segments: {len(segment_counts)}
   • Top Segment: {segment_counts.index[0]} ({segment_counts.values[0]} customers)
   • Average Customer Value: ${rfm['Monetary'].mean():.2f}

📈 FORECAST:
   • Next 30 days predicted revenue: ${forecast['yhat'].iloc[-90:-60].sum():,.2f}
   • Next 90 days predicted revenue: ${forecast['yhat'].tail(90).sum():,.2f}

📁 DELIVERABLES:
   • retail_analytics_complete.xlsx - Ready for Power BI
   • Interactive visualizations above
   • Inventory optimization recommendations
   • Customer segmentation analysis

✅ System ready for business intelligence deployment!
""")

print("\n🚀 Execution completed successfully!")
print("📊 All analytics, forecasts, and exports are ready")