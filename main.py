# regression_streamlit.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Optional: XGBoost
try:
    from xgboost import XGBRegressor
    has_xgb = True
except Exception:
    has_xgb = False

# ------------------------
# Load Data
# ------------------------
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Streamlit UI
# ------------------------
st.title("📊 Regression Models Comparison (Diabetes Dataset)")

st.write("This demo runs different regression models (Linear, RandomForest, XGBoost if available) "
         "on the sklearn diabetes dataset and shows evaluation metrics.")

model_choice = st.sidebar.selectbox(
    "Choose a model",
    ["Linear Regression", "Random Forest"] + (["XGBoost"] if has_xgb else [])
)

# ------------------------
# Build pipelines
# ------------------------
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

if model_choice == "Linear Regression":
    model = Pipeline([('pre', numeric_pipeline),
                      ('model', LinearRegression())])

elif model_choice == "Random Forest":
    model = Pipeline([('imputer', SimpleImputer(strategy='median')),
                      ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

elif model_choice == "XGBoost" and has_xgb:
    model = Pipeline([('imputer', SimpleImputer(strategy='median')),
                      ('model', XGBRegressor(n_estimators=200, random_state=42, tree_method="auto"))])

# ------------------------
# Train & Evaluate
# ------------------------
if st.button("Run Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader(f"Results for {model_choice}")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R² Score", f"{r2:.3f}")

    st.line_chart(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).reset_index(drop=True))
