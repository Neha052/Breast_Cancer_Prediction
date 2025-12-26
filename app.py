import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import pickle 
from sklearn.datasets import load_breast_cancer

# ========== LOAD DATASET ==========
@st.cache_data
def load_dataset():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data


# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    try:
        with open("final_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
# ========== BUILD INPUT FORM ==========
def build_input_form(feature_names):
    st.subheader("Enter feature values")

    user_input = {}
    for col in feature_names:
        user_input[col] = st.number_input(
            label=col,
            value=0.0,
            step=0.1,
            format="%.4f",
        )

    return pd.DataFrame([user_input])
# ========== EDA ==========
def show_eda(df):
    st.subheader("Dataset overview")
    st.write(df.head())

    st.subheader("Summary statistics")
    st.write(df.describe())

    numeric_cols = df.columns[:-1]  # all except target

    st.subheader("Feature distribution")
    feature = st.selectbox("Select a feature", numeric_cols)
    st.line_chart(df[feature])


# ========== PREDICTION ==========
def make_prediction(model, input_df, data):
    st.subheader("Prediction result")
    prediction = model.predict(input_df)[0]

    target_names = data.target_names

    st.markdown(f"**Predicted class:** `{target_names[prediction]}`")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        proba_df = pd.DataFrame({
            "class": target_names,
            "probability": proba
        }).sort_values("probability", ascending=False)

        st.subheader("Probability scores")
        st.dataframe(proba_df)

# ========== MAIN ==========
def main():
    st.title("Breast Cancer Classification Demo")
    st.write("""
        This Streamlit app showcases a machine learning model trained on the **Breast Cancer Wisconsin dataset**  
        using scikit-learn.

        Workflow in the original notebook:
        1. Load dataset (sklearn)
        2. Exploratory Data Analysis (EDA) + scaling
        3. Train–test split
        4. Train multiple models and select best one
        5. Save final model as **.pkl**
    """)

    df, data = load_dataset()
    model = load_model()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ("Project Overview", "EDA", "Predict")
    )

    if page == "Project Overview":
        st.header("Project Overview")
        st.write(f"Dataset shape: {df.shape}")
        st.markdown("""
        - **569 samples**
        - **30 numeric features**
        - Binary classification: `malignant` vs `benign`
        """)

    elif page == "EDA":
        show_eda(df)

    elif page == "Predict":
        if model is None:
            st.error("Model cannot be loaded.")
            return

        input_df = build_input_form(data.feature_names)
        if st.button("Predict"):
            make_prediction(model, input_df, data)


if __name__ == "__main__":
    main()