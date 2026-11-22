
import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "best_model.keras"

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

def predict_lung_opacity(image):
    if image is None:
        return "Please upload an image.", {}

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    resized = cv2.resize(gray, (128, 128))
    normalized = resized.astype(np.float32) / 255.0
    model_input = normalized[np.newaxis, ..., np.newaxis]

    probability = float(
        model.predict(model_input, verbose=0)[0][0]
    )

    prediction = (
        "Lung Opacity"
        if probability >= 0.5
        else "No Lung Opacity"
    )

    scores = {
        "No Lung Opacity": 1.0 - probability,
        "Lung Opacity": probability
    }

    return prediction, scores

demo = gr.Interface(
    fn=predict_lung_opacity,
    inputs=gr.Image(type="numpy", label="Upload Chest X-ray"),
    outputs=[
        gr.Textbox(label="Predicted Class"),
        gr.Label(label="Model Scores")
    ],
    title="Chest X-ray Lung Opacity Detection",
    description=(
        "Educational demonstration only. "
        "Not intended for clinical diagnosis."
    )
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
