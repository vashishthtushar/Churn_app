
# 📊 Churn Prediction Dashboard  
🌐 **Live Demo**: [https://churnapp-kdqdhkxsgsnksuedzyehcv.streamlit.app/](https://churnapp-kdqdhkxsgsnksuedzyehcv.streamlit.app/)

An interactive Streamlit-based web application for predicting customer churn in a telecom dataset. This dashboard allows businesses to upload data, visualize churn risks, predict individual customer churn probability, retrain the model, and explore model insights—all through an intuitive user interface.

---

## 🚀 Features

- 📈 **Dashboard Overview** – Summary metrics, churn risk distribution pie chart, and top risky users.
- 🧾 **Real-Time Churn Prediction** – Input new customer data via a form to get churn probability instantly.
- 🔁 **Model Retraining** – Upload new training/test data to retrain the model on-the-fly.
- 🧠 **Insights & Visualizations** – Confusion matrix, churn distribution, and user churn heatmap.
- 📥 **CSV Export** – Download filtered predictions as CSV.
- 💡 **Multi-page UI** – Cleanly separated navigation tabs (Dashboard, Predict, Retrain, Insights).

---

## 🧠 Technologies & Libraries Used

| Library            | Purpose                                       |
|--------------------|-----------------------------------------------|
| `streamlit`        | Interactive UI and deployment framework       |
| `pandas`           | Data manipulation and CSV handling            |
| `numpy`            | Numerical computing                           |
| `scikit-learn`     | Model training, scaling, encoding, evaluation |
| `matplotlib`       | Visualizations (charts, plots)                |
| `seaborn`          | Confusion matrix heatmaps                     |
| `joblib`           | Model saving/loading                          |
| `pickle`           | Storing column order for form inputs          |

---

## 🗂️ Project Structure

churn_app/
├── app.py # Main Streamlit app
├── train_and_predict.py # Model training script
├── churn_predictions.csv # Sample output predictions
├── telco_train.csv # Training dataset
├── telco_test.csv # Test dataset
├── model.pkl # Trained model
├── scaler.pkl # Scaler used in training
├── label_encoders.pkl # Encoders for categorical columns
├── columns.pkl # Column order used in training
├── requirements.txt # Python dependencies
└── README.md

yaml
Copy code

---

## 🧪 How to Run Locally

### ✅ Prerequisites

- Python 3.8 or higher
- Git installed

### 📦 1. Clone the repository

```bash
git clone https://github.com/vashishthtushar/Churn_app.git
cd Churn_app
📥 2. Install dependencies
bash
Copy code
pip install -r requirements.txt
▶️ 3. Run the Streamlit app
bash
Copy code
streamlit run app.py
Now visit: http://localhost:8501 in your browser.

🌐 Deployment (Streamlit Community Cloud)
Push your full project to a public GitHub repo.

Go to https://streamlit.io/cloud and log in.

Click "New App" → Connect GitHub → Select the repo.

Fill:

Branch: main

File: churn_app/app.py

Deploy!

✅ Make sure you include requirements.txt in your GitHub repo so Streamlit installs all necessary libraries.

📂 Required Files
Ensure these files are present before deploying:

app.py

train_and_predict.py

requirements.txt

telco_train.csv

telco_test.csv

model.pkl, scaler.pkl, label_encoders.pkl, columns.pkl (auto-created after running training)

📽️ Demo
🎯 Live App:
https://churnapp-kdqdhkxsgsnksuedzyehcv.streamlit.app/

Or record your own with tools like Loom or OBS Studio.

✨ Future Enhancements (Creative Ideas)
📌 SHAP-based feature importance explanations

📊 User churn timeline visualization

📬 Email alerts for high-risk churn customers

🧾 Auto-predict from uploaded user batch CSVs

📧 Contact
Made with ❤️ by Tushar Vashishth and Chetan
GitHub: @vashishthtushar

