import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("Starting Data Generation and Model Training...")
np.random.seed(42)
n_customers = 5000

data = {
    'Company_ID': [f"CMP_{i}" for i in range(1000, 1000 + n_customers)],
    'Account_Age_Months': np.random.randint(1, 60, n_customers),
    'Monthly_Recurring_Revenue': np.random.uniform(500, 10000, n_customers),
    'Active_Users': np.random.randint(5, 500, n_customers),
    'Support_Tickets_Last_Month': np.random.randint(0, 15, n_customers),
    'Days_Since_Last_Login': np.random.randint(1, 90, n_customers),
    'Contract_Type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_customers, p=[0.5, 0.3, 0.2])
}

df = pd.DataFrame(data)
churn_prob = (
    (df['Support_Tickets_Last_Month'] > 5).astype(int) * 0.3 +
    (df['Days_Since_Last_Login'] > 30).astype(int) * 0.4 +
    (df['Contract_Type'] == 'Month-to-Month').astype(int) * 0.2 +
    np.random.uniform(0, 0.2, n_customers) 
)

df['Churn'] = (churn_prob > 0.6).astype(int)
df.to_csv('data/client_data.csv', index=False)
print("✅ Synthetic dataset generated and saved to 'data/client_data.csv'")
X = df.drop(columns=['Company_ID', 'Churn'])
X = pd.get_dummies(X, columns=['Contract_Type'], drop_first=True)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=7)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model trained! Test Accuracy: {accuracy * 100:.2f}%")
model_artifacts = {
    'model': model,
    'features': list(X.columns)
}
joblib.dump(model_artifacts, 'models/churn_rf_model.pkl')
print("✅ Model artifacts saved to 'models/churn_rf_model.pkl'")