import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

st.set_page_config(page_title="2025AA05590_MLAssignment2", layout="wide")

st.title("ML Assignment 2 (2025AA05590)")

model_name = st.selectbox(
    "Select Model",
    ["KNN","Decision Tree","Logistic Regression","Naive Bayes","Random Forest"]
)

file = st.file_uploader("Please upload a CSV File", type=["csv"])

if file is None:
    st.info("Upload dataset to begin")
    st.stop()

try:
    df = pd.read_csv(file, sep=None, engine="python")
except:
    st.error("Could not read CSV file.")
    st.stop()

st.subheader("Uploaded Data Preview")
st.dataframe(df.head())

target_col = None

for col in df.columns:
    if col.lower() in ["target","y","label","class"]:
        target_col = col
        break

if target_col is None:
    st.error("No target column found. Expected column name like: Target / y / label")
    st.stop()

st.success(f"Detected Target Column → {target_col}")

y = df[target_col]
X = df.drop(columns=[target_col])

if y.dtype == object:
    y = y.str.lower().map({"yes":1,"no":0,"true":1,"false":0})

if y.isnull().any():
    st.error("Target column must contain only binary values.")
    st.stop()

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier()
}

if st.button("Train Model"):

    model = models[model_name]

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train,y_train)
    preds = pipe.predict(X_test)

   
    acc = accuracy_score(y_test,preds)
    prec = precision_score(y_test,preds)
    rec = recall_score(y_test,preds)
    f1 = f1_score(y_test,preds)

    try:
        probs = pipe.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test,probs)
    except:
        auc = np.nan
        probs = None

    # DISPLAY METRICS
    st.subheader("Model Performance")

    col1,col2,col3,col4,col5 = st.columns(5)

    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")
    col3.metric("Recall", f"{rec:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")
    col5.metric("AUC", "N/A" if np.isnan(auc) else f"{auc:.3f}")

    # CONFUSION MATRIX
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test,preds)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,cm[i,j],ha="center",va="center")

    st.pyplot(fig)

    # ROC CURVE
    if probs is not None:
        st.subheader("ROC Curve")
        fig2, ax2 = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, probs, ax=ax2)
        st.pyplot(fig2)

    st.success("Training Successfully Complete")
