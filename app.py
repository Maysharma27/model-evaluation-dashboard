import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="ML Model Evaluator", layout="wide")

st.title("📊 Machine Learning Classification Model Evaluator")
st.write("Upload test data and select a pre-trained model to evaluate performance.")

# --- SIDEBAR: Upload and Model Selection ---
st.sidebar.header("Configuration")

# Feature a: dataset upload (CSV)
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

# Feature b: model selection dropdown
model_option = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine",
    ],
)

# File names created by train.py (inside the model/ folder)
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl",
}
SCALER_FILE = "model/scaler.pkl"

# These 3 models were trained on SCALED data, so they need the scaler too.
NEEDS_SCALING = {"Logistic Regression", "KNN", "Support Vector Machine"}

target_col = st.sidebar.text_input("Target Column Name", value="target")


@st.cache_resource
def load_model(filepath):
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None


if uploaded_file is None:
    st.info("👈 Please upload a test CSV file from the sidebar to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in the uploaded CSV.")
    st.stop()

X_test = df.drop(columns=[target_col])
y_true = df[target_col]

model_path = MODEL_FILES[model_option]
model = load_model(model_path)
if model is None:
    st.error(f"Model file `{model_path}` not found! Check your `model/` folder.")
    st.stop()

# Scale the data for the models that were trained on scaled data
if model_option in NEEDS_SCALING:
    scaler = load_model(SCALER_FILE)
    if scaler is None:
        st.error(f"Scaler file `{SCALER_FILE}` not found! Check your `model/` folder.")
        st.stop()
    try:
        X_input = scaler.transform(X_test[list(scaler.feature_names_in_)])
    except KeyError:
        st.error(
            "The uploaded CSV does not have the same feature columns "
            "as the training data."
        )
        st.stop()
else:
    X_input = X_test

y_pred = model.predict(X_input)

# ROC AUC
try:
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_input)
        if len(np.unique(y_true)) == 2:
            roc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    elif hasattr(model, "decision_function"):
        roc = roc_auc_score(y_true, model.decision_function(X_input))
    else:
        roc = "N/A"
except Exception:
    roc = "N/A"

# Same definition as train.py: positive class (1) for 2-class data
avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)

st.markdown(f"### Results for **{model_option}**")

# Feature c: evaluation metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("Precision", f"{prec:.4f}")
col3.metric("Recall", f"{rec:.4f}")
col4.metric("F1 Score", f"{f1:.4f}")
col5.metric("AUC", f"{roc:.4f}" if isinstance(roc, float) else str(roc))
col6.metric("MCC", f"{mcc:.4f}")

st.markdown("---")

# Feature d: confusion matrix and classification report
col_cm, col_rep = st.columns(2)

with col_cm:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_rep:
    st.subheader("Classification Report")
    report_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
