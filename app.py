import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Churn Predictor App", layout="wide")

# -------------------- SIDEBAR NAV --------------------
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🧾 Predict", "⚙️ Retrain"])

# -------------------- LOAD MODEL UTILS --------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load("label_encoders.pkl")

@st.cache_resource
def load_columns():
    with open("columns.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()
le_dict = load_encoders()
model_columns = load_columns()

# -------------------- PAGE 1: DASHBOARD --------------------
if page == "📊 Dashboard":
    st.title("📊 Churn Prediction Dashboard")
    df = pd.read_csv("churn_predictions.csv")

    st.markdown("### 📌 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", len(df))
    col2.metric("High Risk Users", (df['churn_probability'] > 0.7).sum())
    col3.metric("Average Probability", f"{df['churn_probability'].mean():.2f}")

    with st.expander("📈 Churn Risk Distribution"):
        sizes = [(df['churn_probability'] > 0.7).sum(), (df['churn_probability'] <= 0.7).sum()]
        labels = ['High Risk (>0.7)', 'Low Risk (<=0.7)']
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🚨 Top Risk Users")
    st.dataframe(df.sort_values("churn_probability", ascending=False).head(10))

    st.markdown("---")
    st.markdown("### 📋 All Users")
    threshold = st.slider("Churn Probability Threshold", 0.0, 1.0, 0.0)
    st.dataframe(df[df["churn_probability"] >= threshold])

    st.download_button("⬇️ Download Filtered CSV", df.to_csv(index=False), "churn_predictions.csv")

# -------------------- PAGE 2: PREDICT --------------------
elif page == "🧾 Predict":
    st.title("🧾 Real-Time Churn Prediction")

    with st.form("user_input_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        SeniorCitizen = 1 if SeniorCitizen == "Yes" else 0
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

        submitted = st.form_submit_button("🚀 Predict Churn")

    if submitted:
        input_data = {
            "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner, "Dependents": Dependents,
            "tenure": tenure, "PhoneService": PhoneService, "MultipleLines": MultipleLines,
            "InternetService": InternetService, "OnlineSecurity": OnlineSecurity, "Contract": Contract,
            "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges, "PaymentMethod": PaymentMethod
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

        st.markdown("### 🔍 Feature Impact")
        st_shap = st.components.v1.html if hasattr(st.components, 'v1') else st.components.html
        shap_html = shap.force_plot(explainer.expected_value[1], shap_values[1], input_df, matplotlib=False, show=False)
        st_shap(shap_html.html(), height=300)

# -------------------- PAGE 3: RETRAIN --------------------
elif page == "⚙️ Retrain":
    st.title("⚙️ Retrain Churn Model")
    uploaded_train = st.file_uploader("Upload Train CSV", type="csv")
    uploaded_test = st.file_uploader("Upload Test CSV", type="csv")

    if st.button("🔁 Retrain Model"):
        if uploaded_train and uploaded_test:
            train_df = pd.read_csv(uploaded_train)
            test_df = pd.read_csv(uploaded_test)
            test_ids = test_df["customerID"]
            train_df.drop("customerID", axis=1, inplace=True)
            test_df.drop("customerID", axis=1, inplace=True)

            train_df["TotalCharges"] = pd.to_numeric(train_df["TotalCharges"], errors="coerce")
            test_df["TotalCharges"] = pd.to_numeric(test_df["TotalCharges"], errors="coerce")
            train_df.fillna(0, inplace=True)
            test_df.fillna(0, inplace=True)

            le_dict = {}
            for col in train_df.columns:
                if train_df[col].dtype == 'object':
                    le = LabelEncoder()
                    train_df[col] = le.fit_transform(train_df[col])
                    test_df[col] = le.transform(test_df[col])
                    le_dict[col] = le

            X_train = train_df.drop("churned", axis=1)
            y_train = train_df["churned"]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            test_scaled = scaler.transform(test_df)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

            with open("model.pkl", "wb") as f: joblib.dump(model, f)
            with open("scaler.pkl", "wb") as f: joblib.dump(scaler, f)
            with open("label_encoders.pkl", "wb") as f: joblib.dump(le_dict, f)
            with open("columns.pkl", "wb") as f: pickle.dump(X_train.columns.tolist(), f)

            churn_probs = model.predict_proba(test_scaled)[:, 1]
            output = pd.DataFrame({"user_id": test_ids, "churn_probability": churn_probs})
            output.to_csv("churn_predictions.csv", index=False)
            st.success("✅ Model retrained and predictions saved!")

            y_pred = model.predict(X_train_scaled)
            cm = confusion_matrix(y_train, y_pred)
            st.markdown("### 📉 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            st.markdown("### 📌 Feature Importances")
            importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            st.bar_chart(importances.head(10))
        else:
            st.error("⚠️ Please upload both train and test datasets to retrain.")
