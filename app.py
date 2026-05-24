import gradio as gr
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load model and label map
model = tf.keras.models.load_model("waste_cnn_best.keras")

with open("label_map.json") as f:
    label_map = json.load(f)  # {"0": "aerosol_cans", ...}

def predict(image):
    # Preprocess — match exactly how you trained
    img = Image.fromarray(image).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Run model
    preds = model.predict(arr)[0]

    # Build top-5 result dict for Gradio Label output
    top5_idx = preds.argsort()[-5:][::-1]
    results = {}
    for i in top5_idx:
        label = label_map[str(i)].replace("_", " ").title()
        results[label] = float(preds[i])

    return results

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload waste image"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
    title="Waste Classifier — Ei Mon Soe",
    description="Upload a photo of a waste item. The CNN model classifies it into one of 30 categories.",
    examples=[],
    allow_flagging="never"
)

demo.launch()