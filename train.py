import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import skops.io as sio
import os

# Créer les dossiers nécessaires
os.makedirs("Model", exist_ok=True)
os.makedirs("Results", exist_ok=True)

# 1. Chargement des données
print("📊 Chargement des données...")
df = pd.read_csv("Data/data.csv")

# Nettoyer les données
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

# Mélanger les données
df = df.sample(frac=1, random_state=125)

print(f"✅ Dataset chargé : {df.shape}")
print(df.head(3))

# 2. Préparation des données
X = df.drop("diagnosis", axis=1).values
y = df.diagnosis.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

# 3. Pipeline de Machine Learning
print("\n🔧 Construction du pipeline...")

num_col = list(range(X.shape[1]))

transform = ColumnTransformer(
    [
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)

# 4. Entraînement
print("\n🎯 Entraînement du modèle...")
pipe.fit(X_train, y_train)

# 5. Évaluation
print("\n📈 Évaluation du modèle...")
predictions = pipe.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, pos_label="M")

print(f"Accuracy: {round(accuracy * 100, 2)}%")
print(f"F1 Score: {round(f1, 2)}")

# 6. Sauvegarde des métriques
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.")

print("✅ Métriques sauvegardées dans Results/metrics.txt")

# 7. Matrice de confusion
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)
plt.close()

print("✅ Matrice de confusion sauvegardée dans Results/model_results.png")

# 8. Sauvegarde du modèle
sio.dump(pipe, "Model/breast_cancer_pipeline.skops")
print("✅ Modèle sauvegardé dans Model/breast_cancer_pipeline.skops")

print("\n🎉 Entraînement terminé avec succès!")
