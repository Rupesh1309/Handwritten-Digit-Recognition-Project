import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from streamlit_drawable_canvas import st_canvas

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

@st.cache_resource
def load_model():
    m = DigitCNN()
    m.load_state_dict(torch.load("digit_model.pth", map_location="cpu"))
    m.eval()
    return m

model = load_model()

def preprocess_canvas(img_array):
    img = Image.fromarray(img_array.astype(np.uint8)).convert("L")
    bbox = img.getbbox()
    if not bbox:
        return None
    img = img.crop(bbox)
    img = ImageOps.expand(img, border=int(max(img.size)*0.2), fill=0)
    img = img.resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr - 33.3) / 78.6
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)

def preprocess_upload(image):
    img = image.convert("L")
    
    img = ImageEnhance.Contrast(img).enhance(2.0)
    
    arr = np.array(img)
    thresh = arr.mean()
    arr = np.where(arr < thresh, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    
    if np.mean(arr) > 127:
        img = ImageOps.invert(img)
    
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img = ImageOps.expand(img, border=int(max(img.size)*0.2), fill=0)
    img = img.resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr - 33.3) / 78.6
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)

def predict(tensor):
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred = probs.argmax().item()
        conf = probs[pred].item()
    return pred, conf, probs.tolist()

st.set_page_config(page_title="Digit Recognition", page_icon="✍️", layout="centered")
st.title("✍️ Handwritten Digit Recognition")
st.caption("Post Office / Bank Form Digit Reader")

tab1, tab2 = st.tabs(["🖊️ Write on Canvas", "📷 Upload Form Photo"])

with tab1:
    st.write("Draw a digit(0-9) below:")
    canvas = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=280, width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    if st.button("🔍 Predict", key="canvas_btn"):
        if canvas.image_data is not None:
            tensor = preprocess_canvas(canvas.image_data)
            if tensor is not None:
                pred, conf, probs = predict(tensor)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Digit", pred)
                with col2:
                    st.metric("Confidence", f"{conf*100:.1f}%")
                if conf < 0.7:
                    st.warning("⚠️ Low confidence — Draw again")
                else:
                    st.success("✅ High confidence prediction!")
                st.write("**All digits probability:**")
                for i, p in enumerate(probs):
                    st.progress(float(p), text=f"Digit {i}: {p*100:.1f}%")
            else:
                st.warning("First Draw Something!")

with tab2:
    st.write("Upload the picture of Bank/Post Office form:")
    st.info("💡 Tips: Please crop and upload only the digit area. Make sure the digit is visible properly.")
    uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded", width=250)
        if st.button("🔍 Predict", key="upload_btn"):
            tensor = preprocess_upload(image)
            pred, conf, probs = predict(tensor)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Digit", pred)
            with col2:
                st.metric("Confidence", f"{conf*100:.1f}%")
            if conf < 0.7:
                st.warning("⚠️ Confidence kam hai — better quality image try karein")
            else:
                st.success("✅ High confidence prediction!")
            st.write("**All digits probability:**")
            for i, p in enumerate(probs):
                st.progress(float(p), text=f"Digit {i}: {p*100:.1f}%")
