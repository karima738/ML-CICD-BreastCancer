import gradio as gr
import pandas as pd
import skops.io as sio

model = sio.load("../Model/breast_cancer_model.skops", trusted=True)

def predict(features):
    df = pd.DataFrame([features], columns=feature_names)
    result = model.predict(df)[0]
    return "Malignant (cancer)" if result == "M" else "Benign (non cancer)"

# Load feature names based on dataset
feature_names = pd.read_csv("../Data/data.csv").drop(columns=["id","Unnamed: 32","diagnosis"], errors="ignore").columns.tolist()

inputs = [gr.Number(label=col) for col in feature_names]

interface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="Breast Cancer Diagnostic Model"
)

interface.launch()
