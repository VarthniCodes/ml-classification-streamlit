import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Machine Learning Classification Models App")

st.write("Upload test dataset and evaluate different ML models.")

# Load models
MODEL_PATH = "model"

models = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_PATH, "Logistic Regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(MODEL_PATH, "Decision Tree.pkl")),
    "kNN": joblib.load(os.path.join(MODEL_PATH, "kNN.pkl")),
    "Naive Bayes": joblib.load(os.path.join(MODEL_PATH, "Naive Bayes.pkl")),
    "Random Forest": joblib.load(os.path.join(MODEL_PATH, "Random Forest.pkl")),
    "XGBoost": joblib.load(os.path.join(MODEL_PATH, "XGBoost.pkl"))
}

scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV Test Data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

    target_column = st.text_input("Enter target column name", "target")

    if target_column in df.columns:
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        X = scaler.transform(X)

        model_name = st.selectbox("Select Model", list(models.keys()))
        model = models[model_name]

        y_pred = model.predict(X)

        st.subheader("Evaluation Metrics")

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.write("Accuracy:", acc)
        st.write("Precision:", prec)
        st.write("Recall:", rec)
        st.write("F1 Score:", f1)
        st.write("MCC Score:", mcc)

        if hasattr(model, "predict_proba"):
            auc = roc_auc_score(y, model.predict_proba(X)[:,1])
            st.write("AUC Score:", auc)

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

    else:
        st.error("Target column not found in dataset.")
