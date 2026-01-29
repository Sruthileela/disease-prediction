<<<<<<< HEAD
# simple_ml.py - Simple Machine Learning Example
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("📊 Creating simple ML model...")

# Create fake medical data (we'll use real data later)
data = {
    'age': [25, 45, 35, 50, 23, 45, 60, 30],
    'bmi': [22.0, 28.5, 24.0, 30.2, 21.5, 27.8, 32.0, 23.5],
    'blood_pressure': [120, 140, 130, 150, 118, 138, 160, 125],
    'diabetes': [0, 1, 0, 1, 0, 1, 1, 0]  # 0=No, 1=Yes
}

# Create DataFrame
df = pd.DataFrame(data)
print("\n📋 Sample Data:")
print(df.head())

# Prepare features and target
X = df[['age', 'bmi', 'blood_pressure']]  # Features
y = df['diabetes']  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\n📈 Training samples: {len(X_train)}")
print(f"📊 Testing samples: {len(X_test)}")

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"\n✅ Model Accuracy: {accuracy:.2%}")

# Make prediction for a new patient
new_patient = [[40, 26.5, 135]]  # Age, BMI, BP
prediction = model.predict(new_patient)
risk = "HIGH RISK" if prediction[0] == 1 else "LOW RISK"
=======
# simple_ml.py - Simple Machine Learning Example
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("📊 Creating simple ML model...")

# Create fake medical data (we'll use real data later)
data = {
    'age': [25, 45, 35, 50, 23, 45, 60, 30],
    'bmi': [22.0, 28.5, 24.0, 30.2, 21.5, 27.8, 32.0, 23.5],
    'blood_pressure': [120, 140, 130, 150, 118, 138, 160, 125],
    'diabetes': [0, 1, 0, 1, 0, 1, 1, 0]  # 0=No, 1=Yes
}

# Create DataFrame
df = pd.DataFrame(data)
print("\n📋 Sample Data:")
print(df.head())

# Prepare features and target
X = df[['age', 'bmi', 'blood_pressure']]  # Features
y = df['diabetes']  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\n📈 Training samples: {len(X_train)}")
print(f"📊 Testing samples: {len(X_test)}")

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"\n✅ Model Accuracy: {accuracy:.2%}")

# Make prediction for a new patient
new_patient = [[40, 26.5, 135]]  # Age, BMI, BP
prediction = model.predict(new_patient)
risk = "HIGH RISK" if prediction[0] == 1 else "LOW RISK"
>>>>>>> b6e68bb28e978c5c868bda60e99b2b56f506f76e
print(f"\n🩺 New Patient Prediction: {risk}")