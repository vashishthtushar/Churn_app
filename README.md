# 📊 Churn Prediction Dashboard  
🌐 **Live Demo**: [https://churnapp-kdqdhkxsgsnksuedzyehcv.streamlit.app/](https://churnapp-kdqdhkxsgsnksuedzyehcv.streamlit.app/)

An interactive Streamlit-based web application for predicting customer churn in a telecom dataset. Explore client data, retrain models, and visualize insights—all via a user-friendly interface.

---

## 🚀 Available Features

### 📈 Dashboard
- **Summary Metrics** – Total users, high-risk count, and average churn probability cards.
- **Pie Chart** – High vs. low churn risk breakdown.
- **Top Risk Table** – Lists top 10 customers by churn probability.
- **Filtered View** – Adjust churn-probability threshold to filter users.
- **Download CSV** – Export filtered results easily.

### 🧾 Real-Time Prediction
- **User Input Form** – Input user attributes (tenure, services, payment method, etc.) and get churn probability instantly.
- **Risk-level Advice** – Automatically display "High", "Medium", or "Low" risk suggestions.
- **SHAP Feature Impact** – Interactive SHAP visualization showing which features most influenced the individual prediction.

### ⚙️ Retrain Model (UI)
- **File Uploads** – Upload new training and test CSVs directly in app.
- **Single-click Retrain & Predict** – Automatically retrains with uploaded data and outputs new predictions.
- **Performance Summary** – Confusion matrix and feature importance chart displayed immediately post-train.
- **Auto-Update Dashboard** – Retrained predictions populate the Dashboard with fresh analysis.

---

## 🧠 Prediction Pipeline Overview

1. **Data Input & Preprocessing**  
   - Handles missing values, encodes categorical feats via `LabelEncoder`, scales numeric with `StandardScaler`.

2. **Model Architecture**  
   - RandomForestClassifier (100 trees, `random_state=42`) for churn classification.

3. **Explanation & Thresholding**  
   - Uses probability output for risk classification (threshold = 0.7), accompanied by SHAP explanation.

4. **Retraining Mechanism**  
   - Seamlessly retrains the same pipeline upon data upload and visualizes new model performance.

---

## 📈 Model Performance

- **Algorithm**: Random Forest Classifier
- **Train-Test Split**: 80-20
- **Metric Used**: ROC AUC Score

| Metric        | Value     |
|---------------|-----------|
| Accuracy      | ~83%      |
| ROC AUC Score | ~0.86     |
| Precision     | Balanced  |
| Recall        | Good churn detection sensitivity |

> 📌 These values may vary slightly after retraining using new data.


## 📊 Built-in Visualizations

- Pie chart (risk distribution)
- Top 10 churners bar
- SHAP summary plot (feature influence per prediction)
- Confusion matrix and feature importance visuals after retraining

---

## 🧩 Tech Stack & Dependencies

| Package            | Purpose                                       |
|--------------------|-----------------------------------------------|
| `streamlit`        | Web UI framework                              |
| `pandas`, `numpy`  | Data handling and numerical operations        |
| `scikit-learn`     | Model, scaling, encoding, and evaluation      |
| `matplotlib`, `seaborn` | Plotting, confusion matrix                |
| `joblib`, `pickle` | Save/load model, encoders, column metadata    |
| `shap`             | Interpretability via explainable AI           |

---

## 🚀 Deployment Setup

**Prerequisites**:
- Python 3.8+, Git, streamlit CLI

**Install & Run Locally**:
```bash
git clone https://github.com/vashishthtushar/Churn_app.git
cd Churn_app
pip install -r requirements.txt
streamlit run app.py
```

- In future we can enhance more this application:

Future Enhancements
* Live batch uploading of user data for prediction

* Automated alerting for flagging high-risk users (email/SMS)

* Interactive dashboards (e.g., retention trends over time)

* Integration with CRM systems

Made with ❤️ by Tushar Vashishth and Chetan
GitHub: @vashishthtushar
