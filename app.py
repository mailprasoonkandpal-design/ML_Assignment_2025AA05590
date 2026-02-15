import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="2025AA05590", layout="centered")

st.title("ML Assignment 2 (2025AA05590) — Classification Models")

# ---------------- MODEL LIST ----------------
models = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

model_name = st.selectbox("Select Model", list(models.keys()))

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# ---------------- MAIN ----------------
if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")

        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        # Detect target column automatically
        target_col = None

        for col in df.columns[::-1]:
            if df[col].dtype == "object" or df[col].nunique() <= 2:
                target_col = col
                break

        if target_col is None:
            st.error("❌ Could not detect target column")
            st.stop()

        st.info(f"Detected Target Column → **{target_col}**")

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # convert yes/no → 1/0 if needed
        if y.dtype == "object":
            y = y.astype(str).str.lower().map({"yes":1,"no":0})

        # Load model
        model_path = models[model_name]

        if not os.path.exists(model_path):
            st.error("Model file not found. Train model first.")
            st.stop()

        model = load_model(model_path)

        # Predict
        y_pred = model.predict(X)

        #st.subheader("Predictions")
        #st.write(y_pred[:20])

        # Metrics if target exists
        if y.isnull().sum() == 0:

            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                matthews_corrcoef,
                confusion_matrix
            )

            st.subheader("Evaluation Metrics")

            st.write("Accuracy:", accuracy_score(y, y_pred))
            st.write("Precision:", precision_score(y, y_pred))
            st.write("Recall:", recall_score(y, y_pred))
            st.write("F1 Score:", f1_score(y, y_pred))
            st.write("MCC:", matthews_corrcoef(y, y_pred))

            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y, y_pred))

        else:
            st.warning("Target column contains null values. Metrics skipped.")

    except Exception as e:
        st.error("App Error — Please check dataset format.")
        st.exception(e)

else:
    st.info("Upload a dataset to start predictions.")
