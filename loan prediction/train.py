import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

print("⏳ Generating dataset and training model...")

# 1. Create a dummy dataset matching your Streamlit inputs
np.random.seed(42)
n_samples = 500

data = {
    "Age": np.random.randint(18, 70, size=n_samples),
    "Gender": np.random.choice(["Male", "Female"], size=n_samples),
    "Qualification": np.random.choice(["Graduate", "Not Graduate"], size=n_samples),
    "Annual_Income": np.random.randint(20000, 150000, size=n_samples),
    "Loan_Amount": np.random.randint(5000, 50000, size=n_samples),
    "Credit_Score": np.random.randint(300, 850, size=n_samples),
    "Dependents": np.random.randint(0, 5, size=n_samples),
    "Property_Value": np.random.randint(10000, 300000, size=n_samples),
    "Gold_Grams": np.random.randint(0, 100, size=n_samples)
}

df = pd.DataFrame(data)

# Logical target variable for dummy data
df["Loan_Status"] = np.where(
    (df["Credit_Score"] > 600) & (df["Annual_Income"] > df["Loan_Amount"] * 1.5), 1, 0
)

# 2. Save CSV just in case
df.to_csv("loan_dataset.csv", index=False)

# 3. Encode categorical variables
label_encoders = {}
for col in ["Gender", "Qualification"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. Define features (X) and target (y)
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

# 5. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 6. Save binary pickle files
with open("loan.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("🎉 Success! loan.pkl and label_encoders.pkl have been created!")