import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import pickle
import shap
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Churn App", layout="wide")

# ------------------------ Utility Functions --------------------------
@st.cache_data
def load_predictions():
    if os.path.exists("churn_predictions.csv"):
        return pd.read_csv("churn_predictions.csv")
    else:
        return pd.DataFrame()

def load_model_components():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_dict = joblib.load("label_encoders.pkl")
    with open("columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
    return model, scaler, le_dict, model_columns

def encode_features(df, le_dict):
    for col in df.columns:
        if df[col].dtype == 'object':
            if col in le_dict:
                le = le_dict[col]
                df[col] = le.transform(df[col])
    return df

def retrain_model(train_df, test_df):
    le_dict = {}

    train_df["TotalCharges"] = pd.to_numeric(train_df["TotalCharges"], errors="coerce")
    test_df["TotalCharges"] = pd.to_numeric(test_df["TotalCharges"], errors="coerce")
    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)

    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])
            test_df[col] = le.transform(test_df[col])
            le_dict[col] = le

    X = train_df.drop(columns=["churned"])
    y = train_df["churned"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    test_scaled = scaler.transform(test_df)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    preds = model.predict_proba(test_scaled)[:, 1]

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le_dict, "label_encoders.pkl")
    with open("columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    return preds, test_df, y, model

# ------------------------ Navigation --------------------------
pages = ["Dashboard", "Predict", "Retrain"]
selection = st.sidebar.radio("Go to", pages)

# ------------------------ Dashboard --------------------------
if selection == "Dashboard":
    st.title("📊 Churn Prediction Dashboard")
    df = load_predictions()

    if df.empty:
        st.warning("No predictions available. Please retrain model first.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users", len(df))
        col2.metric("High Risk Users", (df['churn_probability'] > 0.7).sum())
        col3.metric("Avg Probability", f"{df['churn_probability'].mean():.2f}")

        labels = ['High Risk (>0.7)', 'Low Risk (<=0.7)']
        sizes = [(df['churn_probability'] > 0.7).sum(), (df['churn_probability'] <= 0.7).sum()]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%')
        st.pyplot(fig)

        st.subheader("🚨 Top Risk Users")
        st.dataframe(df.sort_values("churn_probability", ascending=False).head(10))

        st.subheader("📋 All Users")
        threshold = st.slider("Churn Probability Threshold", 0.0, 1.0, 0.0)
        st.dataframe(df[df["churn_probability"] >= threshold])

        st.download_button("Download Filtered CSV", df.to_csv(index=False), "churn_predictions.csv")

# ------------------------ Prediction --------------------------
elif selection == "Predict":
    st.title("🧾 Real-Time Churn Prediction")
    try:
        model, scaler, le_dict, model_columns = load_model_components()

        with st.form("user_input_form"):
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = 1 if st.selectbox("Senior Citizen", ["No", "Yes"]) == "Yes" else 0
            Partner = st.selectbox("Has Partner", ["Yes", "No"])
            Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            MonthlyCharges = st.number_input("Monthly Charges", value=50.0)
            TotalCharges = st.number_input("Total Charges", value=600.0)
            PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

            submitted = st.form_submit_button("Predict Churn")
            if submitted:
                input_data = {
                    "gender": gender,
                    "SeniorCitizen": SeniorCitizen,
                    "Partner": Partner,
                    "Dependents": Dependents,
                    "tenure": tenure,
                    "PhoneService": PhoneService,
                    "MultipleLines": MultipleLines,
                    "InternetService": InternetService,
                    "OnlineSecurity": OnlineSecurity,
                    "Contract": Contract,
                    "MonthlyCharges": MonthlyCharges,
                    "TotalCharges": TotalCharges,
                    "PaymentMethod": PaymentMethod
                }

                for key in input_data:
                    if key != "SeniorCitizen" and isinstance(input_data[key], str):
                        input_data[key] = le_dict[key].transform([input_data[key]])[0]

                input_df = pd.DataFrame([input_data])
                for col in model_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[model_columns]
                input_scaled = scaler.transform(input_df)
                prob = model.predict_proba(input_scaled)[0][1]

                st.success(f"🔥 Churn Probability: **{prob:.2f}**")

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_scaled)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader("🔍 Feature Impact")
                shap.initjs()
                shap.force_plot(explainer.expected_value[1], shap_values[1], input_df, matplotlib=True, show=False)
                st.pyplot(bbox_inches='tight')

    except Exception as e:
        st.error(f"Error loading model components: {e}")

# ------------------------ Retraining --------------------------
elif selection == "Retrain":
    st.title("🛠 Retrain Model")
    st.markdown("Upload a new train and test CSV to train a fresh model.")

    train_file = st.file_uploader("Upload Training CSV", type=["csv"], key="train")
    test_file = st.file_uploader("Upload Test CSV", type=["csv"], key="test")

    if train_file and test_file:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        if "customerID" in train_df.columns:
            train_df.drop(columns=["customerID"], inplace=True)
        if "customerID" in test_df.columns:
            user_ids = test_df["customerID"]
            test_df.drop(columns=["customerID"], inplace=True)
        else:
            user_ids = pd.Series(np.arange(len(test_df)), name="user_id")

        if "churned" not in train_df.columns:
            st.error("'churned' column missing in training data.")
        else:
            churn_probs, test_df_encoded, y, model = retrain_model(train_df, test_df)
            df_out = pd.DataFrame({
                "user_id": user_ids,
                "churn_probability": churn_probs
            })
            df_out.to_csv("churn_predictions.csv", index=False)
            st.success("✅ Retraining complete and predictions saved!")

            # Display confusion matrix on training data
            X_train = train_df.drop(columns=["churned"])
            y_train = train_df["churned"]
            scaler = joblib.load("scaler.pkl")
            X_train_scaled = scaler.transform(X_train)
            y_pred_train = model.predict(X_train_scaled)

            cm = confusion_matrix(y_train, y_pred_train)
            st.subheader("Confusion Matrix on Training Data")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

            st.subheader("Feature Importances")
            importances = pd.Series(model.feature_importances_, index=X_train.columns)
            st.bar_chart(importances.sort_values(ascending=False).head(10))
