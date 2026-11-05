"""train.py - trains 5 classification models on the Adult Income dataset.

Run once:   python train.py
It creates:  model/*.pkl, dataset.csv, test_data.csv, model_comparison.csv
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

os.makedirs("model", exist_ok=True)

# Step 1: load the raw dataset
df = pd.read_csv("adult.csv")

# Step 2: target column (1 for >50K, 0 for <=50K)
df["target"] = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)
df = df.drop(columns=["income"])

# Step 3: remove missing values ('?') and convert TEXT columns to numbers.
# Only text columns are converted - numeric columns (age, hours, ...) stay as they are.
df = df.replace("?", np.nan).dropna()
text_columns = df.select_dtypes(exclude="number").columns
for col in text_columns:
    df[col] = df[col].astype("category").cat.codes

df.to_csv("dataset.csv", index=False)

# Step 4: 80/20 train-test split
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Small CSV you can upload in the Streamlit app
test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv("test_data.csv", index=False)

# Step 5: feature scaling (needed by Logistic Regression, KNN and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "model/scaler.pkl")

# Step 6: train all 5 models and calculate the 6 metrics
# name -> (model, file name, needs scaled data?)
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42),
                            "logistic_regression.pkl", True),
    "Decision Tree": (DecisionTreeClassifier(random_state=42, max_depth=10),
                      "decision_tree.pkl", False),
    "KNN": (KNeighborsClassifier(n_neighbors=5), "knn.pkl", True),
    "Naive Bayes": (GaussianNB(), "naive_bayes.pkl", False),
    "Support Vector Machine": (SVC(probability=True, random_state=42),
                               "svm.pkl", True),
}

rows = []
for name, (model, filename, needs_scaling) in models.items():
    print(f"Training {name} ...")
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled if needs_scaling else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    rows.append({
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    })
    joblib.dump(model, f"model/{filename}")

results = pd.DataFrame(rows)
results.to_csv("model_comparison.csv", index=False)
print()
print(results.to_string(index=False))
