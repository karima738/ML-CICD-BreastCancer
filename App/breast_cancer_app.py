import gradio as gr
import pandas as pd
import skops.io as sio

# Charger le modèle avec tous les types nécessaires
untrusted_types = sio.get_untrusted_types(file="Model/breast_cancer_pipeline.skops")
pipe = sio.load("Model/breast_cancer_pipeline.skops", trusted=untrusted_types)

# Charger les noms des features
df = pd.read_csv("Data/data.csv")
feature_names = df.drop(columns=["id", "Unnamed: 32", "diagnosis"], errors="ignore").columns.tolist()


def predict_cancer(*features):
    """
    Prédire si une tumeur est maligne ou bénigne.
    """
    df_input = pd.DataFrame([features], columns=feature_names)
    prediction = pipe.predict(df_input)[0]

    if prediction == "M":
        return "🔴 Malignant (Cancerous)"
    else:
        return "🟢 Benign (Non-cancerous)"


# Créer les inputs
inputs = [
    gr.Slider(minimum=0, maximum=50, step=0.1, label=name, value=0.0)
    for name in feature_names
]

outputs = gr.Textbox(label="Diagnostic Prediction")

# Exemples
examples = [
    [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
     1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
     25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
    [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
     0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
     15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259],
]

title = "🏥 Breast Cancer Classification"
description = """
Enter the cell nucleus characteristics to predict if a tumor is **malignant** or **benign**.

⚠️ **Disclaimer**: This is a demo model for educational purposes only. Always consult healthcare professionals.
"""

article = """
This app is part of the CI/CD for Machine Learning guide.
It demonstrates automated training, evaluation, and deployment using GitHub Actions.
"""

# Interface Gradio
gr.Interface(
    fn=predict_cancer,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()