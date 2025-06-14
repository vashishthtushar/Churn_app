import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
import pickle

# Load data
train_df = pd.read_csv("telco_train.csv")
test_df = pd.read_csv("telco_test.csv")

# Save test user IDs
test_ids = test_df["customerID"]

# Drop customerID from train/test
train_df = train_df.drop(columns=["customerID"])
test_df = test_df.drop(columns=["customerID"])

# Convert TotalCharges to numeric
train_df["TotalCharges"] = pd.to_numeric(train_df["TotalCharges"], errors="coerce")
test_df["TotalCharges"] = pd.to_numeric(test_df["TotalCharges"], errors="coerce")

# Fill missing values
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# Encode categorical features
le_dict = {}
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        le_dict[col] = le

# Split features and label
X = train_df.drop(columns=["churned"])
y = train_df["churned"]

# ✅ Save column names BEFORE converting to NumPy array
column_names = X.columns.tolist()

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_df_scaled = scaler.transform(test_df)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Predict probabilities
churn_probs = model.predict_proba(test_df_scaled)[:, 1]

# Save predictions to CSV
output = pd.DataFrame({
    "user_id": test_ids,
    "churn_probability": churn_probs
})
output.to_csv("churn_predictions.csv", index=False)
print("✅ churn_predictions.csv saved!")

# ✅ Save model and processors
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_dict, "label_encoders.pkl")

# ✅ Save column order
with open("columns.pkl", "wb") as f:
    pickle.dump(column_names, f)

print("✅ Model, encoders, scaler, and column structure saved.")
