import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_model(MLproject2\model):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Missing model file: {mlassignment2/MLproject2}")
        st.stop()
        
# -----------------------------
# 1️ Load Saved Models
lr = joblib.load("model/logistic.pkl")
dt = joblib.load("model/decision_tree.pkl")
knn = joblib.load("model/knn.pkl")
nb = joblib.load("model/naive_bayes.pkl")
rf = joblib.load("model/random_forest.pkl")
xgb = joblib.load("model/xgboost.pkl")
scaler = joblib.load("model/scaler.pkl")

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "KNN": knn,
    "Naive Bayes": nb,
    "Random Forest": rf,
    "XGBoost": xgb
}

# -----------------------------
# 2️ App Title
st.title("Heart Disease Prediction App")
st.write("Upload a CSV file and select a model to predict heart disease.")

# -----------------------------
# 3️ CSV Upload
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # -----------------------------
    # 4️ Model Selection Dropdown
    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    # -----------------------------
    # 5️ Prediction
    if st.button("Predict"):
        # Scale features for scaled models
        if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
            X_input = scaler.transform(df)
        else:
            X_input = df.values

        y_pred = model.predict(X_input)

        # Try to get probabilities for AUC
        try:
            y_prob = model.predict_proba(X_input)[:, 1]
        except:
            y_prob = y_pred  # fallback

        # -----------------------------
        # 6️ Display Evaluation Metrics (if actual target exists in uploaded CSV)
        if 'target' in df.columns:
            y_true = df['target']
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

            st.subheader("Evaluation Metrics")
            st.write(f"**Accuracy:** {acc:.3f}")
            st.write(f"**AUC:** {auc:.3f}")
            st.write(f"**Precision:** {prec:.3f}")
            st.write(f"**Recall:** {rec:.3f}")
            st.write(f"**F1 Score:** {f1:.3f}")
            st.write(f"**MCC Score:** {mcc:.3f}")

            # -----------------------------
            # 7️ Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        else:
            st.warning("No 'target' column in CSV. Predictions only will be shown.")
            st.subheader("Predictions")
            st.write(y_pred)
