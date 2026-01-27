# main.py - Our main program
print("🎉 Welcome to Disease Prediction Project!")
print("This is our first Python file!")

# Simple calculation
age = 45
print(f"Age: {age} years")

# Simple list
diseases = ["Diabetes", "Heart Disease", "Stroke"]
print(f"Diseases to predict: {diseases}")

# main.py - Updated
from src.data_loader import load_sample_data

print("=== Disease Prediction System ===")
print("Loading data...")

# Load data
df = load_sample_data()
print(f"\nData loaded: {len(df)} patients")
print(df.head())

print("\n✅ Project structure complete!")
print("\nNext steps:")
print("1. Get real medical datasets")
print("2. Add more diseases")
print("3. Create web interface")
print("4. Upload to GitHub")