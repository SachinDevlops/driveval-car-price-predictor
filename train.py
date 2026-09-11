import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# 1. Load dataset
df = pd.read_csv('car_data.csv')

# 2. Feature engineering: Extract brand name
df['brand'] = df['name'].apply(lambda x: x.split()[0])

# 3. Select relevant features and target
features = ['brand', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
X = df[features]
y = df['selling_price']

# 4. One-Hot Encode categorical variables
categorical_cols = ['brand', 'fuel', 'seller_type', 'transmission', 'owner']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# 6. Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 7. Model evaluation
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"Mean Absolute Error: ₹ {mean_absolute_error(y_test, y_pred):,.2f}")

# 8. Export artifacts
joblib.dump(model, 'car_model.pkl')
joblib.dump(list(X_encoded.columns), 'model_columns.pkl')
joblib.dump(sorted(df['brand'].unique()), 'brands.pkl')
print("Model, columns, and brand lists saved successfully!")