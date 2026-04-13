%%writefile app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

save_dir = '/content/drive/MyDrive/Handwritten_Digit_Recognition'

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

model_path = os.path.join(save_dir, 'model.pth')
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

st.title("✍️ Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    st.success(f"Prediction: {pred.item()} ({confidence.item()*100:.2f}%) ")
