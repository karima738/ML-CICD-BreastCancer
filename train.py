import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import skops.io as sio

# Load dataset
df = pd.read_csv("Data/data.csv")

# Clean dataset
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

# Separate features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ML pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=200))
])

# Train
pipe.fit(X_train, y_train)

# Predict
pred = pipe.predict(X_test)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="macro")

# Save metrics
with open("Results/metrics.txt", "w") as f:
    f.write(f"Accuracy={acc:.4f}, F1={f1:.4f}")

# Save confusion matrix
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("Results/confusion_matrix.png")

# Save model
sio.dump(pipe, "Model/breast_cancer_model.skops")
