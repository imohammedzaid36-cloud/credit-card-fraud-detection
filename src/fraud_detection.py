import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("dataset/creditcard.csv")
print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())
print(data["Class"].value_counts())
sns.countplot(x="Class", data=data)
plt.show()

print(data["Class"].value_counts())
X = data.drop("Class", axis=1)

y = data["Class"]
X = data.drop("Class", axis=1)
y = data["Class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Data split completed!")

from sklearn.ensemble import RandomForestClassifier

print("Creating model...")

model = RandomForestClassifier(
    n_estimators=10,
    random_state=42
)

print("Training model...")

model.fit(X_train, y_train)

print("Training completed!")

y_pred = model.predict(X_test)

print("Prediction completed!")
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred
    )
)
sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Confusion Matrix")
plt.show()
import joblib

joblib.dump(
    model,
    "fraud_model.pkl"
)