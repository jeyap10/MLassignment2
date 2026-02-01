import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2 - Classification",
    layout="wide"
)
st.title(" Name: Jeya Prakash S") 
 st.title("Bits id: 2024dc04016")
st.title(" ML Assignment 2 – Classification Models")
st.write("Upload test dataset under the github path MLproject2/data you can download the excel file and upload it, select model, and view evaluation metrics.")

# --------------------------------------------------
# Safe Model Loader
# --------------------------------------------------
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f" Model file missing: {path}")
        st.stop()

# --------------------------------------------------
# Load Models
# --------------------------------------------------
lr = load_model("MLproject2/model/logistic.pkl")
dt = load_model("MLproject2/model/decision_tree.pkl")
knn = load_model("MLproject2/model/knn.pkl")
nb = load_model("MLproject2/model/naive_bayes.pkl")
rf = load_model("MLproject2/model/random_forest.pkl")
xgb = load_model("MLproject2/model/xgboost.pkl")

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "KNN": knn,
    "Naive Bayes": nb,
    "Random Forest": rf,
    "XGBoost": xgb
}

# --------------------------------------------------
# Upload Dataset
# --------------------------------------------------
st.header(" Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader(" Dataset Preview")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error(" Dataset must contain a 'target' column")
        st.stop()

    X = df.drop(columns=["target"])
    y_true = df["target"]

    # --------------------------------------------------
    # Model Selection
    # --------------------------------------------------
    st.subheader(" Select Model")
    model_name = st.selectbox("Choose a classification model", list(models.keys()))
    model = models[model_name]

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    if st.button("Final Predict and Evaluate"):

        y_pred = model.predict(X.values)

        # Some models support predict_proba
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X.values)[:, 1]
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = "N/A"

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted")
        rec = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")
        mcc = matthews_corrcoef(y_true, y_pred)

        st.subheader(" Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(acc, 4))
        col2.metric("Precision", round(prec, 4))
        col3.metric("Recall", round(rec, 4))

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", round(f1, 4))
        col5.metric("MCC", round(mcc, 4))
        col6.metric("AUC Score", auc if auc == "N/A" else round(auc, 4))

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------
        st.subheader(" Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        st.pyplot(fig)

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------
        st.subheader(" Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
