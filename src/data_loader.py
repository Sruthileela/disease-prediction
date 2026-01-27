# src/data_loader.py
import pandas as pd

def load_sample_data():
    """Load sample medical data"""
    data = {
        'patient_id': [1, 2, 3, 4, 5],
        'age': [34, 45, 28, 56, 39],
        'glucose': [85, 140, 90, 200, 120],
        'bmi': [22.1, 30.5, 21.0, 35.2, 28.7],
        'has_diabetes': [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)