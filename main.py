import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def diabetes_prediction():
    data = pd.read_csv("data/diabetes.csv")
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Diabetes Accuracy:", accuracy_score(y_test, y_pred))

    sample = [X.iloc[0].tolist()]
    result = model.predict(sample)
    print("Diabetes Prediction:", "YES" if result[0] == 1 else "NO")


def heart_prediction():
    data = pd.read_csv("data/heart.csv")
    X = data.drop("result", axis=1)
    y = data["result"]
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Heart Disease Accuracy:", accuracy_score(y_test, y_pred))

    sample = [X.iloc[0].tolist()]
    result = model.predict(sample)
    print("Heart Disease Prediction:", "YES" if result[0] == 1 else "NO")


def stroke_prediction():
    data = pd.read_csv("data/stroke.csv")
    data = data.dropna()

    X = data.drop("stroke", axis=1)
    y = data["stroke"]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Stroke Accuracy:", accuracy_score(y_test, y_pred))

    sample = [X.iloc[0].tolist()]
    result = model.predict(sample)
    print("Stroke Prediction:", "YES" if result[0] == 1 else "NO")


print("Select Disease")
print("1. Diabetes")
print("2. Heart Disease")
print("3. Stroke")

choice = input("Enter choice (1/2/3): ")

if choice == "1":
    diabetes_prediction()
elif choice == "2":
    heart_prediction()
elif choice == "3":
    stroke_prediction()
else:
    print("Invalid choice")
   
=======
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def diabetes_prediction():
    data = pd.read_csv("data/diabetes.csv")
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Diabetes Accuracy:", accuracy_score(y_test, y_pred))

    sample = [X.iloc[0].tolist()]
    result = model.predict(sample)
    print("Diabetes Prediction:", "YES" if result[0] == 1 else "NO")


def heart_prediction():
    data = pd.read_csv("data/heart.csv")
    X = data.drop("result", axis=1)
    y = data["result"]
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Heart Disease Accuracy:", accuracy_score(y_test, y_pred))

    sample = [X.iloc[0].tolist()]
    result = model.predict(sample)
    print("Heart Disease Prediction:", "YES" if result[0] == 1 else "NO")


def stroke_prediction():
    data = pd.read_csv("data/stroke.csv")
    data = data.dropna()

    X = data.drop("stroke", axis=1)
    y = data["stroke"]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Stroke Accuracy:", accuracy_score(y_test, y_pred))

    sample = [X.iloc[0].tolist()]
    result = model.predict(sample)
    print("Stroke Prediction:", "YES" if result[0] == 1 else "NO")


print("Select Disease")
print("1. Diabetes")
print("2. Heart Disease")
print("3. Stroke")

choice = input("Enter choice (1/2/3): ")

if choice == "1":
    diabetes_prediction()
elif choice == "2":
    heart_prediction()
elif choice == "3":
    stroke_prediction()
else:
    print("Invalid choice")
>>>>>>> 706805378f0a4b9fbb281e42e876d7f8605c6596
