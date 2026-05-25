import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score



st.set_page_config(
    page_title="California Housing Price Prediction",
    layout="wide"
)

st.title("Housing Price Prediction")
st.write("Predict Median House Value using Linear Regression")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("housing.csv")
    return df

df = load_data()

# DATA PREVIEW
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.write("Shape:", df.shape)

# DATA CLEANING
df = df.dropna()

# Target column
TARGET = "price"

# Convert categorical column
df = pd.get_dummies(df, drop_first=True)

# FEATURES & TARGET

X = df.drop(TARGET, axis=1)
y = df[TARGET]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL TRAINING

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)



# EVALUATION
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader(" Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
col4.metric("R² Score", f"{r2:.4f}")

# ---------------------------
# USER INPUT SECTION
# ---------------------------
st.subheader(" Predict House Price")

median_income = st.number_input(
    "Median Income",
    min_value=0.0,
    value=3.5
)

housing_median_age = st.number_input(
    "Housing Median Age",
    min_value=1,
    value=25
)

total_rooms = st.number_input(
    "Total Rooms",
    min_value=1,
    value=2000
)

total_bedrooms = st.number_input(
    "Total Bedrooms",
    min_value=1,
    value=400
)

population = st.number_input(
    "Population",
    min_value=1,
    value=1000
)

households = st.number_input(
    "Households",
    min_value=1,
    value=350
)

latitude = st.number_input(
    "Latitude",
    value=34.0
)

longitude = st.number_input(
    "Longitude",
    value=-118.0
)

# ---------------------------
# CREATE INPUT DATAFRAME
# ---------------------------
input_data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income]
})

# Add missing dummy columns
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[X.columns]

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Median House Value: ${prediction:,.2f}"
    )

# ---------------------------
# FEATURE IMPORTANCE
# ---------------------------
st.subheader("Feature Coefficients")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coef_df = coef_df.sort_values(
    by="Coefficient",
    ascending=False
)

st.dataframe(coef_df)