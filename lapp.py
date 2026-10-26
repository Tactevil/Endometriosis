import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
import segmentation_models_pytorch as smp
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="Endometriosis AI Assistant", layout="wide", page_icon="🩺", initial_sidebar_state="expanded")

# --- PATHS ---
BASE_DIR = Path(r"C:\Users\Admin\Endometriosis\glenda_results")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .hero-title { font-size: 38px; font-weight: bold; color: #1A5276; margin-bottom: 20px; }
    .insight-box { background-color: #e8f4f8; padding: 20px; border-left: 6px solid #2980b9; border-radius: 8px; margin-top: 15px; font-size: 16px; color: #333333 !important; }
    
    /* FORCED DARK TEXT FOR ALERTS */
    .warning-box { background-color: #fce4e4; padding: 20px; border-left: 6px solid #c0392b; border-radius: 8px; margin-top: 15px; font-size: 16px; color: #333333 !important; }
    .success-box { background-color: #e8f8f5; padding: 20px; border-left: 6px solid #27ae60; border-radius: 8px; margin-top: 15px; font-size: 16px; color: #333333 !important; }
    
    .metric-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; border-top: 5px solid #2980b9; }
    .metric-value { font-size: 28px; font-weight: bold; color: #2980b9; }
    .metric-label { font-size: 16px; color: #7f8c8d; }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    
    # 1. Load ResNet-50 (Detection)
    resnet = timm.create_model('resnet50', pretrained=False, num_classes=1)
    resnet_path = BASE_DIR / "resnet50_rq1_comparison.pth"
    if resnet_path.exists():
        resnet.load_state_dict(torch.load(resnet_path, map_location=device))
    else:
        st.error("⚠️ resnet50_rq1_comparison.pth not found in the results folder.")
    resnet.eval()
    
    # 2. Load U-Net Tversky (Localization)
    unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    unet_path = BASE_DIR / "unet_rq5_tversky_v2.pth"
    if unet_path.exists():
        unet.load_state_dict(torch.load(unet_path, map_location=device))
    else:
        st.error("⚠️ unet_rq5_tversky_v2.pth not found in the results folder.")
    unet.eval()
    
    return resnet, unet, device

resnet_model, unet_model, device = load_models()

# --- TRANSFORMS ---
cls_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
seg_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- SIDEBAR ---
st.sidebar.title("🩺 Endometriosis AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Project Overview", "🔬 Live Clinical Demo"]
)

# ==========================================
# PAGE 1: PROJECT OVERVIEW
# ==========================================
if page == "🏠 Project Overview":
    st.markdown('<div class="hero-title">AI-Assisted Endometriosis Detection</div>', unsafe_allow_html=True)
    st.write("A deep learning pipeline for lesion detection and localisation in laparoscopic images (GLENDA v1.5).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### The Clinical Problem
        Endometriosis affects ~10% of women. During laparoscopy, surgeons must manually scan for lesions that vary in color, size, and are often obscured by blood, smoke, or glare.
        
        ### Our Solution
        We trained deep learning models to act as a real-time assistant:
        *   **Step 1: Classification.** A ResNet-50 model analyzes the frame to determine if it contains pathology (Endometriosis vs. Healthy).
        *   **Step 2: Segmentation.** If pathology is detected, a Tversky U-Net draws pixel-perfect boundaries around the suspected lesions.
        """)
    with col2:
        st.markdown("""
        ### The Dataset
        *   **Source:** GLENDA v1.5 (University of Klagenfurt)
        *   **Annotated Frames:** 373 (from 102 surgeries)
        *   **Healthy Frames:** 13,438
        *   **Lesion Types:** Peritoneum, Ovary, Uterus, and TIE (Deep Infiltrating Endometriosis)
        
        ### Model Performance (Test Set)
        *   **ResNet-50 AUROC:** 0.8948 (Excellent detection capability)
        *   **U-Net Dice Score:** 0.3780 (Strong localisation for tiny, diffuse lesions)
        """)
        
    st.markdown('<div class="insight-box"><b>Why this matters:</b> By automating the initial detection and outlining of lesions, surgeons can reduce missed lesions and focus their attention on surgical excision.</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 2: LIVE CLINICAL DEMO
# ==========================================
elif page == "🔬 Live Clinical Demo":
    st.markdown('<div class="hero-title">Live Lesion Detection & Segmentation</div>', unsafe_allow_html=True)
    st.write("Upload a laparoscopic image to run real-time inference using the trained models.")
    
    uploaded_file = st.file_uploader("Upload a Laparoscopic Image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Surgical Frame", width="stretch")
            
        # 1. Run Classification (ResNet-50)
        img_tensor_cls = cls_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_output = resnet_model(img_tensor_cls)
            pathology_prob = torch.sigmoid(cls_output).item()
            
        # 2. Run Segmentation (Tversky U-Net)
        img_tensor_seg = seg_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            seg_output = unet_model(img_tensor_seg)
            mask_prob = torch.sigmoid(seg_output).squeeze().cpu().numpy()
            binary_mask = (mask_prob > 0.5).astype(np.uint8)
            
        # Resize mask back to original image size
        mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(image.size, Image.NEAREST)
        
        with col2:
            # Create Overlay
            if pathology_prob > 0.5:
                # Blend a red mask over the lesion
                red_overlay = Image.new("RGB", image.size, (255, 0, 0))
                overlay = Image.composite(red_overlay, image, mask_pil)
                overlay = Image.blend(image, overlay, alpha=0.4)
                st.image(overlay, caption="U-Net Localisation (Red = Lesion)", width="stretch")
            else:
                st.image(image, caption="No Lesion Predicted", width="stretch")

        st.markdown("---")
        
        # Results Area
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("### 🧠 Detection Result (ResNet-50)")
            if pathology_prob > 0.5:
                st.markdown(f'<div class="warning-box"><b>⚠️ ENDOMETRIOSIS DETECTED</b><br>Confidence: {pathology_prob*100:.1f}%<br><br>The AI has identified features consistent with a lesion.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-box"><b>✅ HEALTHY TISSUE</b><br>Confidence: {(1-pathology_prob)*100:.1f}%<br><br>The AI did not detect significant pathological features.</div>', unsafe_allow_html=True)
                
        with res_col2:
            st.markdown("### 🎯 Localisation Output (Tversky U-Net)")
            lesion_pixels = np.sum(binary_mask)
            total_pixels = binary_mask.size
            coverage = (lesion_pixels / total_pixels) * 100
            
            st.markdown(f"""
            - **Lesion Coverage:** {coverage:.2f}% of the frame
            - **Pixels Segmented:** {lesion_pixels} pixels
            - **Architecture:** U-Net (ResNet-34 Encoder)
            """)
            if lesion_pixels == 0:
                st.warning("The U-Net did not find any localized lesion pixels above the 50% probability threshold.")
            else:
                st.info("The red overlay highlights the exact regions the U-Net identified as endometriosis.")

st.sidebar.markdown("---")
st.sidebar.info("⚠️ **Disclaimer:** This is a research prototype trained on the GLENDA v1.5 dataset. It is not intended for clinical diagnostic use.")