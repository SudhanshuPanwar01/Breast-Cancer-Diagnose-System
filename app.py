import os
import gradio as gr
import numpy as np
import tensorflow as tf
import joblib

# =====================================================
# Developed By : Parth
# Roll No      : 241504
# Course       : BCA - Data Science (3rd Year)
# College      : Panipat Institute of Engineering & Technology, Samalkha
# =====================================================

# Load Model
model = tf.keras.models.load_model("breast_cancer_model.h5")

# Load Scaler
scaler = joblib.load("breast_cancer_scaler.pkl")

# -----------------------------------------------------
# Feature Names
# -----------------------------------------------------

feature_names = [
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Smoothness",
    "Mean Compactness",
    "Mean Concavity",
    "Mean Concave Points",
    "Mean Symmetry",
    "Mean Fractal Dimension",
    "Radius Error",
    "Texture Error",
    "Perimeter Error",
    "Area Error",
    "Smoothness Error",
    "Compactness Error",
    "Concavity Error",
    "Concave Points Error",
    "Symmetry Error",
    "Fractal Dimension Error",
    "Worst Radius",
    "Worst Texture",
    "Worst Perimeter",
    "Worst Area",
    "Worst Smoothness",
    "Worst Compactness",
    "Worst Concavity",
    "Worst Concave Points",
    "Worst Symmetry",
    "Worst Fractal Dimension"
]

# -----------------------------------------------------
# Prediction Function
# -----------------------------------------------------

def predict(*inputs):

    data = np.array(inputs).reshape(1, -1)

    data = scaler.transform(data)

    prediction = model.predict(data, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        result = "🔴 Malignant (Cancer Detected)"
        confidence = probability * 100
        color = "#ff4d4d"

    else:
        result = "🟢 Benign (No Cancer Detected)"
        confidence = (1 - probability) * 100
        color = "#16a34a"

    return f"""
    <div style="
    background:{color};
    color:white;
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
    box-shadow:0px 5px 20px rgba(0,0,0,0.3);
    ">

    {result}

    <br><br>

    Confidence : {confidence:.2f}%

    </div>
    """

# -----------------------------------------------------
# Custom CSS
# -----------------------------------------------------

css = """

.gradio-container{
    max-width:1200px !important;
}

footer{
display:none !important;
}

h1{
text-align:center;
}

"""

# -----------------------------------------------------
# Build Interface
# -----------------------------------------------------

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🩺 Breast Cancer Prediction System

### Deep Learning Based Breast Cancer Detection

Enter all 30 medical measurements and click **Predict**.

This system predicts whether the tumor is **Benign** or **Malignant**.
""")

    inputs = []

    with gr.Row():

        with gr.Column():

            for feature in feature_names:
                box = gr.Number(
                    label=feature,
                    value=0
                )
                inputs.append(box)

        with gr.Column():

            output = gr.HTML(label="Prediction")

            predict_btn = gr.Button(
                "🔍 Predict",
                variant="primary",
                size="lg"
            )

            clear_btn = gr.ClearButton(
                components=inputs,
                value="Clear"
            )

    predict_btn.click(
        fn=predict,
        inputs=inputs,
        outputs=output
    )

    gr.Markdown("""

---

## 👨‍💻 Developed By

### **Sudhanshu**

**Roll No : 241533**

**Course : BCA - Data Science (3rd Year)**

**College : Panipat Institute of Engineering & Technology, Samalkha**

---

© 2026 Breast Cancer Prediction System

""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
