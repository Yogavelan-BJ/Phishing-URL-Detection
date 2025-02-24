# /C:/Users/yogav/Desktop/Projects/Phishing-URL-Detection/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
data = pd.read_csv('phishing.csv')

# Print column names to debug
print(data.columns)

# Preprocess data
# ...existing code...

# Split data into training and testing sets
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save model
joblib.dump(model, 'models/phishing_model.pkl')