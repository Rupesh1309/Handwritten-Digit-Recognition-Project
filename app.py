import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Same model class
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64*7*7, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model():
    model = DigitCNN()
    model.load_state_dict(torch.load("digit_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw any digit in the below box.")

# Drawing canvas
canvas = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    height=280, width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas.image_data is not None:
    img = canvas.image_data.astype(np.uint8)
    img = Image.fromarray(img).convert("L")

    if img.getbbox():  # kuch draw hua hai
        img = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array - 33.3) / 78.6  # MNIST normalize
        tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            pred = probs.argmax().item()
            conf = probs[pred].item()

        st.markdown(f"## 🎯 Prediction: **{pred}**")
        st.progress(conf, text=f"Confidence: {conf*100:.1f}%")

        st.write("**Sabhi digits ki probability:**")
        for i, p in enumerate(probs.tolist()):
            st.progress(p, text=f"{i}: {p*100:.1f}%")
