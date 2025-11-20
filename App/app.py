import gradio as gr
import pandas as pd
import skops.io as sio

# ✅ Chemins LOCAUX (fichiers copiés dans App/)
MODEL_PATH = "breast_cancer_model.skops"
DATA_PATH = "data.csv"

# Charger le modèle
try:
    model = sio.load(MODEL_PATH, trusted=True)
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    model = None

# Charger les noms de features
try:
    df = pd.read_csv(DATA_PATH)
    feature_names = df.drop(columns=["id", "Unnamed: 32", "diagnosis"], errors="ignore").columns.tolist()
    print(f"✅ {len(feature_names)} features chargées")
except Exception as e:
    print(f"❌ Erreur chargement features: {e}")
    # Fallback: noms manuels
    feature_names = [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity",
        "mean concave points", "mean symmetry", "mean fractal dimension",
        "radius error", "texture error", "perimeter error", "area error",
        "smoothness error", "compactness error", "concavity error",
        "concave points error", "symmetry error", "fractal dimension error",
        "worst radius", "worst texture", "worst perimeter", "worst area",
        "worst smoothness", "worst compactness", "worst concavity",
        "worst concave points", "worst symmetry", "worst fractal dimension"
    ]


def predict(*features):
    """Prédire si tumeur maligne ou bénigne"""
    if model is None:
        return "❌ Erreur: Modèle non disponible"

    try:
        df = pd.DataFrame([features], columns=feature_names)
        result = model.predict(df)[0]
        return "🔴 Malignant (cancer)" if result == "M" else "🟢 Benign (non-cancer)"
    except Exception as e:
        return f"❌ Erreur: {str(e)}"


# Interface Gradio
inputs = [gr.Number(label=col, value=0.0) for col in feature_names]

interface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Textbox(label="Diagnostic"),
    title="🏥 Breast Cancer Diagnostic Model",
    description="Enter tumor characteristics to predict malignancy",
    theme="soft",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()