import streamlit as st
import os
import time
import json
import requests
from tensorflow_serving.apis.predict_pb2 import PredictRequest
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


request_json = json.dumps({
    "signature_name": "serving_default",
    "instances": None,
})
server_url="http://localhost:8501/v1/models/my_cmapss_model:predict"
#response = requests.post(server_url, data=request_json)
#response.raise_for_status()
#response = response.json()




st.set_page_config(page_title="CSV Explorer", layout="wide")
st.title("📊 CSV Explorer")
st.markdown("Upload a CSV file to explore your data.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
COL_NAMES = (['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] +
             [f'sensor_{i+1}' for i in range(21)])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, names=COL_NAMES)

    st.subheader("📋 Data Preview")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribution", "🔗 Correlation", "📊 Bar Chart", "🧮 Statistics"])

    # --- Tab 1: Distribution ---
    with tab1:
        st.subheader("Distribution of a Numeric Column")
        if numeric_cols:
            col = st.selectbox("Select a numeric column", numeric_cols, key="dist_col")
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
        else:
            st.info("No numeric columns found in the dataset.")

    # --- Tab 2: Correlation Heatmap ---
    with tab2:
        st.subheader("Correlation Heatmap")
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(8, 5))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
        else:
            st.info("Need at least 2 numeric columns to show a correlation heatmap.")

    # --- Tab 3: Bar Chart ---
    with tab3:
        st.subheader("Bar Chart")
        if categorical_cols and numeric_cols:
            cat_col = st.selectbox("Select a categorical column (X axis)", categorical_cols, key="bar_cat")
            num_col = st.selectbox("Select a numeric column (Y axis)", numeric_cols, key="bar_num")
            top_n = st.slider("Number of top categories to show", 3, 20, 10)

            bar_data = df.groupby(cat_col)[num_col].mean().nlargest(top_n).reset_index()
            fig, ax = plt.subplots()
            sns.barplot(data=bar_data, x=cat_col, y=num_col, palette="viridis", ax=ax)
            ax.set_xlabel(cat_col)
            ax.set_ylabel(f"Mean of {num_col}")
            ax.set_title(f"Top {top_n} {cat_col} by {num_col}")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
        elif not categorical_cols:
            st.info("No categorical columns found for the X axis.")
        else:
            st.info("No numeric columns found for the Y axis.")

    # --- Tab 4: Statistics ---
    with tab4:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        st.markdown("#### Missing Values")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]
        missing["% Missing"] = (missing["Missing Values"] / len(df) * 100).round(2)
        st.dataframe(missing, use_container_width=True)

        st.markdown("#### Data Types")
        dtypes = df.dtypes.reset_index()
        dtypes.columns = ["Column", "Data Type"]
        st.dataframe(dtypes, use_container_width=True)

else:
    st.info("👆 Upload a CSV file to get started.")