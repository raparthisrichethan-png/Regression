import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="KNN Insurance Cost Prediction",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# LOAD DATASET
# ==================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

csv_path = BASE_DIR / "data" / "insurance.csv"

df = pd.read_csv(csv_path)
# ==================================================
# LOAD MODEL
# ==================================================

try:
    model = pickle.load(
        open("models/knn_model.pkl", "rb")
    )

    scaler = pickle.load(
        open("models/scaler.pkl", "rb")
    )

except:
    model = None
    scaler = None

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("📊 KNN Regressor")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Overview",
        "EDA",
        "Correlation Analysis",
        "Prediction"
    ]
)

# ==================================================
# HOME
# ==================================================

if page == "Home":

    st.title("💰 Insurance Cost Prediction using KNN Regression")

    st.markdown("""
    ### Project Overview

    This project predicts medical insurance charges using the
    K-Nearest Neighbors (KNN) Regression algorithm.

    ### Features

    ✅ Dataset Overview

    ✅ Exploratory Data Analysis (EDA)

    ✅ Correlation Analysis

    ✅ Interactive Prediction

    ### Target Variable

    **charges**

    ### Algorithm Used

    **KNN Regressor**
    """)

# ==================================================
# DATASET OVERVIEW
# ==================================================

elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    st.subheader("First 5 Records")

    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Datatype": df.dtypes.astype(str)
    })

    st.dataframe(info_df)

    st.subheader("Missing Values")

    st.dataframe(
        pd.DataFrame(
            df.isnull().sum(),
            columns=["Missing Values"]
        )
    )

    st.subheader("Duplicate Records")

    st.write(df.duplicated().sum())

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

# ==================================================
# EDA
# ==================================================

elif page == "EDA":

    st.title("📈 Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    # Histogram

    st.subheader("Histogram")

    feature = st.selectbox(
        "Select Numerical Feature",
        numeric_cols
    )

    fig, ax = plt.subplots()

    sns.histplot(
        df[feature],
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {feature}"
    )

    st.pyplot(fig)

    # Boxplot

    st.subheader("Box Plot")

    box_feature = st.selectbox(
        "Select Feature for Boxplot",
        numeric_cols,
        key="box"
    )

    fig, ax = plt.subplots()

    sns.boxplot(
        x=df[box_feature],
        ax=ax
    )

    st.pyplot(fig)

    # Countplot

    st.subheader("Categorical Analysis")

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    cat_feature = st.selectbox(
        "Select Category",
        categorical_cols
    )

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x=cat_feature,
        ax=ax
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # Charges Distribution

    st.subheader("Charges Distribution")

    fig, ax = plt.subplots()

    sns.histplot(
        df["charges"],
        kde=True,
        ax=ax
    )

    ax.set_title(
        "Insurance Charges Distribution"
    )

    st.pyplot(fig)

# ==================================================
# CORRELATION ANALYSIS
# ==================================================

elif page == "Correlation Analysis":

    st.title("🔥 Correlation Analysis")

    df_corr = df.copy()

    df_corr["sex"] = df_corr["sex"].map({
        "male": 1,
        "female": 0
    })

    df_corr["smoker"] = df_corr["smoker"].map({
        "yes": 1,
        "no": 0
    })

    region_map = {
        "southwest": 0,
        "southeast": 1,
        "northwest": 2,
        "northeast": 3
    }

    df_corr["region"] = (
        df_corr["region"]
        .map(region_map)
    )

    corr = df_corr.corr()

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader(
        "Correlation with Charges"
    )

    st.dataframe(
        corr["charges"]
        .sort_values(
            ascending=False
        )
        .to_frame()
    )

# ==================================================
# PREDICTION
# ==================================================

elif page == "Prediction":

    st.title("💰 Insurance Cost Prediction")

    if model is None:

        st.error(
            "Model file not found. Run notebook and save model first."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                18,
                100,
                25
            )

            sex = st.selectbox(
                "Sex",
                [
                    "male",
                    "female"
                ]
            )

            bmi = st.number_input(
                "BMI",
                10.0,
                60.0,
                25.0
            )

        with col2:

            children = st.number_input(
                "Children",
                0,
                10,
                0
            )

            smoker = st.selectbox(
                "Smoker",
                [
                    "yes",
                    "no"
                ]
            )

            region = st.selectbox(
                "Region",
                [
                    "southwest",
                    "southeast",
                    "northwest",
                    "northeast"
                ]
            )

        if st.button("Predict Insurance Cost"):

            sex = 1 if sex == "male" else 0

            smoker = 1 if smoker == "yes" else 0

            region_map = {
                "southwest": 0,
                "southeast": 1,
                "northwest": 2,
                "northeast": 3
            }

            region = region_map[region]

            input_data = pd.DataFrame(
                [[
                    age,
                    sex,
                    bmi,
                    children,
                    smoker,
                    region
                ]],
                columns=[
                    "age",
                    "sex",
                    "bmi",
                    "children",
                    "smoker",
                    "region"
                ]
            )

            input_scaled = scaler.transform(
                input_data
            )

            prediction = model.predict(
                input_scaled
            )

            st.success(
                f"Estimated Insurance Cost: ${prediction[0]:,.2f}"
            )
