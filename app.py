import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(page_title="AI Image Detector", layout="centered")

st.title("🧠 AI vs Real Image Detector")
st.write("Upload an image to detect whether it is real or AI-generated.")

st.sidebar.title("📌 About")
st.sidebar.write("Final Year Project")
st.sidebar.write("Model: ResNet18 + Grad-CAM")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# =========================
# CORRECT TRANSFORM (NO AUGMENTATION)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# =========================
# PREDICTION + GRADCAM
# =========================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Center image (nice UI)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, caption="Uploaded Image", width=150)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

        prob = torch.nn.functional.softmax(output, dim=1)
        confidence = prob[0][pred].item()

    # =========================
    # RESULT
    # =========================
    if pred.item() == 0:
        st.error("❌ AI Generated Image")
    else:
        st.success("✅ Real Image")

    st.info(f"Confidence: {confidence * 100:.2f}%")

    # =========================
    # GRAD-CAM
    # =========================
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    img_np = np.array(image.resize((224, 224))) / 255.0
    grayscale_cam = cam(input_tensor=img)[0]

    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    st.subheader("🔥 Model Attention (Grad-CAM)")
    st.image(cam_image, use_column_width=True)