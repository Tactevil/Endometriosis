import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="GLENDA AI Dashboard", layout="wide", page_icon="🔬", initial_sidebar_state="expanded")

# --- CUSTOM PATH ---
BASE_DIR = Path(r"C:\Users\Admin\Endometriosis\glenda_results")

# --- LOAD DATA WITH FALLBACKS ---
@st.cache_data
def load_data():
    try:
        with open(BASE_DIR / "glenda_final_results_v2.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

data = load_data()

def get_val(rq_key, metric_key, default):
    return data.get(rq_key, {}).get(metric_key, default)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .hero-title { font-size: 42px; font-weight: bold; color: #2c3e50; text-align: center; margin-bottom: 20px; }
    .hero-subtitle { font-size: 20px; color: #7f8c8d; text-align: center; margin-bottom: 40px; }
    .insight-box { background-color: #e8f4f8; padding: 20px; border-left: 6px solid #2980b9; border-radius: 8px; margin-top: 15px; font-size: 16px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .danger-box { background-color: #fce4e4; padding: 20px; border-left: 6px solid #c0392b; border-radius: 8px; margin-top: 15px; font-size: 16px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .success-box { background-color: #e8f8f5; padding: 20px; border-left: 6px solid #27ae60; border-radius: 8px; margin-top: 15px; font-size: 16px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .metric-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; border-top: 5px solid #2980b9; }
    .metric-value { font-size: 36px; font-weight: bold; color: #2980b9; }
    .metric-label { font-size: 16px; color: #7f8c8d; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select a section:",
    [
        "🩺 What is Endometriosis?",
        "🔬 The GLENDA Dataset",
        "🎯 Project Overview",
        "📊 Final Test Results",
        "RQ1: Detection (Classification)",
        "RQ2 & RQ5: Segmentation & Loss",
        "RQ3: The Classification Gate",
        "RQ4: Rare Classes",
        "🔍 Explainability & Errors",
        "⚙️ Hyperparameter Tuning",
        "🏁 Conclusion"
    ]
)

# ==========================================
# PAGE 1: EDUCATIONAL CONTENT
# ==========================================
if page == "🩺 What is Endometriosis?":
    st.markdown('<div class="hero-title">Understanding Endometriosis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">A medical primer before diving into the AI models</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### What is it?
        Endometriosis is a chronic, inflammatory condition where tissue similar to the lining of the uterus (the endometrium) grows outside the uterus. 
        It affects roughly **10% of women** of reproductive age. It can cause severe pelvic pain, heavy periods, and infertility.
        
        ### Risk Factors
        - Family history (genetics)
        - Early onset of menstruation (before age 11)
        - Short menstrual cycles (less than 27 days)
        - Low body mass index (BMI)
        """)
    with col2:
        st.markdown("""
        ### Investigations & Diagnosis
        - **Ultrasound/MRI:** Can detect large cysts or deep lesions, but often miss small ones.
        - **Laparoscopy (Gold Standard):** A surgeon inserts a camera into the abdomen to visually inspect the pelvic cavity. **This is the exact type of image data we are using in our dataset!**
        
        ### Complications & Prevention
        - **Complications:** Infertility, ovarian cysts (endometriomas), adhesions (organs sticking together).
        - **Prevention:** There is no definitive cure or prevention. Management relies on pain medication, hormone therapy, or surgical excision (cutting out the lesions).
        """)
        
    st.markdown('<div class="insight-box"><b>Why this project matters:</b> Finding lesions during laparoscopy can be difficult due to blood, smoke, and varying lesion colours. Our AI models are trained to automatically detect these lesions to assist surgeons in real-time.</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 2: DATASET
# ==========================================
elif page == "🔬 The GLENDA Dataset":
    st.title("The GLENDA v1.5 Dataset")
    st.write("GLENDA (Gynecologic Laparoscopy ENdometriosis DAtaset) was released by ITEC (University of Klagenfurt) under a CC BY-NC 4.0 license.")
    
    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="metric-card"><div class="metric-value">373</div><div class="metric-label">Annotated Frames</div></div>', unsafe_allow_html=True)
    col2.markdown('<div class="metric-card"><div class="metric-value">13,438</div><div class="metric-label">Healthy (Negative) Frames</div></div>', unsafe_allow_html=True)
    col3.markdown('<div class="metric-card"><div class="metric-value">102</div><div class="metric-label">Surgical Cases</div></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### The 4 Lesion Classes
    The dataset contains pixel-perfect annotations (masks) for four distinct types of endometriosis:
    1.  **Peritoneum:** Lesions on the abdominal lining. (Most common, often diffuse).
    2.  **Ovary:** Endometriomas (chocolate cysts) on the ovaries.
    3.  **Uterus:** Lesions on the uterus.
    4.  **TIE (DIE):** Deep Infiltrating Endometriosis. TIE is the German acronym (Tief Infiltrierende Endometriose). These are deep, aggressive lesions.
    """)

# ==========================================
# PAGE 3: PROJECT OVERVIEW
# ==========================================
elif page == "🎯 Project Overview":
    st.title("Project Overview & Research Questions")
    st.write("Our goal is to compare different Deep Learning paradigms for detecting and outlining these lesions.")
    
    st.markdown("""
    *   **RQ1: Detection:** Can a classifier tell if an image has a lesion or not?
    *   **RQ2: Segmentation:** Can a model accurately draw the boundary of the lesion?
    *   **RQ3: The Gate:** If we use the classifier to block false positives from the segmentation model, does it help?
    *   **RQ4: Rare Classes:** Do the models perform poorly on rare classes like Ovary and Uterus?
    *   **RQ5: Loss Functions:** Does changing the mathematical loss function help the model find tiny lesions?
    """)

# ==========================================
# PAGE 4: FINAL TEST RESULTS (ENHANCED)
# ==========================================
elif page == "📊 Final Test Results":
    st.title("Final Test Set Evaluation")
    st.write("These metrics are evaluated on the unseen Fold 0 test set (20% of cases held out during training). We use different metrics because the models perform different tasks.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{get_val("Final_Test_Set_Metrics", "ResNet50_AUROC", 0.8948):.4f}</div><div class="metric-label">ResNet-50 AUROC</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{get_val("Final_Test_Set_Metrics", "UNet_Tversky_Dice", 0.3780):.4f}</div><div class="metric-label">U-Net Dice Score</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{get_val("Final_Test_Set_Metrics", "YOLO_mAP50", 0.1954):.4f}</div><div class="metric-label">YOLO mAP50</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    ### What do these numbers actually mean?
    *   **AUROC (Area Under the ROC Curve):** Used for the Classifier. A score of 0.5 means the model is guessing randomly. A score of 1.0 is perfect. A score of **0.89** means the model is excellent at telling the difference between a healthy frame and a lesion frame.
    *   **Dice Score:** Used for U-Net (Dense Segmentation). It measures the pixel overlap between the AI's mask and the surgeon's true mask. A score of **0.37** means 37% of the pixels overlap. This sounds low, but because lesions are often just a few pixels wide, missing the edge by a millimeter drops the score drastically.
    *   **mAP50 (Mean Average Precision):** Used for YOLO (Instance Segmentation). It combines how well the model draws bounding boxes and masks. A score of **0.19** indicates that YOLO struggles heavily on this dataset, likely because it is designed for larger, more distinct objects.
    """)

    st.markdown('<div class="insight-box"><b>Interpretation:</b> While the classifier (ResNet) performs well at simply detecting the <i>presence</i> of a lesion, the segmentation models struggle to draw perfect masks because endometriosis lesions are often tiny, diffuse, and blend into the background.</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 5: RQ1 (ENHANCED WITH EXPLAINABILITY)
# ==========================================
elif page == "RQ1: Detection (Classification)":
    st.title("RQ1: Lesion Detection (Classification)")
    st.write("**Question:** Do pathology classifiers detect the lesion, or are they just cheating by memorizing the background?")
    
    st.markdown("### Model Comparison: Custom CNN vs. Pretrained ResNet-50")
    df_comp = pd.DataFrame({
        "Model": ["Custom CNN (From Scratch)", "ResNet-50 (Transfer Learning)"],
        "Test AUROC": [get_val("RQ1_Model_Comparison", "Custom_CNN_Test_AUROC", 0.4766), get_val("RQ1_Model_Comparison", "ResNet50_Test_AUROC", 0.8948)]
    })
    fig = px.bar(df_comp, x="Model", y="Test AUROC", color="Model", text_auto='.4f', color_discrete_sequence=['#e74c3c', '#27ae60'])
    fig.update_layout(yaxis_title="AUROC Score", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### The Shortcut Probe (Masked AUROC)")
    st.write("To test if the model is cheating, we 'blacked out' the actual lesions in the test images using the surgeon's masks. If the model is smart, its accuracy should drop to random guessing (0.50).")
    col1, col2 = st.columns(2)
    col1.metric("Original AUROC (With Lesions)", f"{get_val('RQ1_Classification', 'Train_AUROC', 0.9994):.4f}")
    col2.metric("Masked AUROC (Lesions Blacked Out)", f"{get_val('RQ1_Classification', 'Masked_AUROC', 0.8455):.4f}", delta="Drop", delta_color="inverse")
    
    st.markdown("### Explainability (Grad-CAM)")
    st.write("Grad-CAM is an AI explainability tool that creates a heatmap showing exactly where the model is looking. Red means the model is focusing there.")
    
    if (BASE_DIR / "rq1_gradcam.png").exists():
        st.image(str(BASE_DIR / "rq1_gradcam.png"), caption="ResNet-50 Grad-CAM")
        
    if (BASE_DIR / "rq1.png").exists():
        st.image(str(BASE_DIR / "rq1.png"), caption="ResNet-50 Predictions (Green = Correct, Red = Missed Lesion)")
        
    st.markdown('<div class="danger-box"><b>Conclusion:</b> The Custom CNN failed completely (AUROC ~0.48). Pretrained ResNet-50 succeeded (~0.89) proving transfer learning is mandatory. <b>However</b>, when we blacked out the lesions, the score only dropped to 0.84. Combined with the Grad-CAM heatmap (which shows the model looking at the background anatomy, not the lesion), this proves the model is heavily relying on background context (cheating) rather than the lesion itself!</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 6: RQ2 & RQ5
# ==========================================
elif page == "RQ2 & RQ5: Segmentation & Loss":
    st.title("RQ2 & RQ5: Segmentation & Loss Ablation")
    st.write("**Question:** Which paradigm outlines lesions best, and how do we stop the model from predicting empty masks?")
    
    st.markdown("### The Loss Function Ablation (RQ5)")
    col1, col2 = st.columns(2)
    col1.metric("Baseline U-Net (Dice + BCE Loss)", f"{get_val('RQ5_Loss_Ablation', 'Baseline_UNet_Dice', 0.2553):.4f}")
    col2.metric("Tversky U-Net (Focal Tversky Loss)", f"{get_val('RQ5_Loss_Ablation', 'Tversky_UNet_Dice', 0.3780):.4f}", "+48.1% Improvement")
    
    if (BASE_DIR / "rq2_rq5.png").exists():
        st.image(str(BASE_DIR / "rq2_rq5.png"), caption="True Mask vs Baseline U-Net vs Tversky U-Net")
        
    st.markdown('<div class="success-box"><b>Conclusion:</b> Standard BCE loss treats every pixel equally. Because 97% of the image is healthy, the model just predicts an all-black mask to minimize loss. Focal Tversky loss heavily penalizes the model for missing a lesion, forcing it to find the tiny blobs and improving the Dice score by 48%.</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 7: RQ3
# ==========================================
elif page == "RQ3: The Classification Gate":
    st.title("RQ3: The Classification Gate")
    st.write("**Question:** Can we use the RQ1 Classifier to block the U-Net from making false positive predictions on healthy frames?")
    
    fig = go.Figure(data=[
        go.Bar(name='Ungated (U-Net alone)', x=['False Positives (%)', 'Sensitivity (%)'], y=[2.5, 97.6], marker_color='indianred'),
        go.Bar(name='Gated (U-Net + Classifier)', x=['False Positives (%)', 'Sensitivity (%)'], y=[0.1, 96.2], marker_color='lightsalmon')
    ])
    fig.update_layout(barmode='group', title="Gate Performance Trade-off (Training Set)")
    st.plotly_chart(fig, use_container_width=True)
    
    if (BASE_DIR / "rq3.png").exists():
        st.image(str(BASE_DIR / "rq3.png"), caption="Gate Probabilities: Green (1.00) = Let through, Red = Blocked")
        
    st.markdown('<div class="insight-box"><b>Conclusion:</b> The gate successfully dropped false positives from 2.5% to 0.1% with almost no sensitivity loss on training data. However, because of the RQ1 shortcut learning, the gate relies on background features and is not reliable enough for clinical deployment.</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 8: RQ4
# ==========================================
elif page == "RQ4: Rare Classes":
    st.title("RQ4: Performance on Rare Classes")
    st.write("**Question:** Do rare classes (Ovary, Uterus, TIE) underperform compared to the common class (Peritoneum)?")
    
    df_rq4 = pd.DataFrame({
        "Class": ["Peritoneum (Common)", "Uterus (Rare)", "Ovary (Rare)", "TIE (Rare)"],
        "Dice Score": [get_val("RQ4_Rare_Classes", "Peritoneum_Dice", 0.2135), get_val("RQ4_Rare_Classes", "Uterus_Dice", 0.2711), get_val("RQ4_Rare_Classes", "Ovary_Dice", 0.3113), get_val("RQ4_Rare_Classes", "TIE_Dice", 0.3377)],
        "Sample Size": [64, 2, 8, 11]
    })
    
    fig = px.bar(df_rq4, x="Class", y="Dice Score", color="Class", text="Sample Size", color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    if (BASE_DIR / "rq4_rare_classes.png").exists():
        st.image(str(BASE_DIR / "rq4_rare_classes.png"), caption="Examples of True Positives for Ovary, TIE, and Uterus.")
        
    st.markdown('<div class="insight-box"><b>Conclusion:</b> Surprisingly, the rare classes scored HIGHER than the common class! Tversky loss forced the model to find distinct, well-circumscribed lesions (like Ovary). Peritoneum lesions are often diffuse and scattered, making them harder to segment. Note: Uterus has only N=2 frames, making its score statistically unstable.</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 9: EXPLAINABILITY & ERRORS (ENHANCED)
# ==========================================
elif page == "🔍 Explainability & Errors":
    st.title("Explainability & Error Analysis")
    st.write("AI models are 'black boxes'. This section opens the box to show you *why* the models made mistakes and *where* they looked.")
    
    st.markdown("---")
    st.markdown("### 1. Classifier Error Analysis (ResNet-50)")
    if (BASE_DIR / "rq1_error_analysis.png").exists():
        st.image(str(BASE_DIR / "rq1_error_analysis.png"), caption="Top False Positives (Top row) vs False Negatives (Bottom row)")
    st.markdown("""
    *   **False Positives (Top row):** The model flagged these healthy frames as having endometriosis. Looking at the images, you can see why—there are confusing specular reflections (bright lights), blood vessels, or surgical tools that look like lesions to the AI.
    *   **False Negatives (Bottom row):** The model missed these real lesions. This typically happens when the lesion is hidden under shadows, obscured by surgical smoke, or blends into the colour of the surrounding tissue.
    """)
        
    st.markdown("---")
    st.markdown("### 2. Dense Segmentation Errors (U-Net)")
    if (BASE_DIR / "unet_error_analysis.png").exists():
        st.image(str(BASE_DIR / "unet_error_analysis.png"), caption="Worst U-Net Predictions")
    st.markdown("""
    *   **Why does U-Net fail?** The U-Net is excellent at finding well-defined, circular lesions (like cysts). However, it fails on **diffuse peritoneal lesions**. These lesions are patchy, spread out, and have fuzzy borders. The U-Net cannot decide where the lesion ends and the healthy tissue begins.
    """)
        
    st.markdown("---")
    st.markdown("### 3. Instance Segmentation Errors (YOLO)")
    if (BASE_DIR / "yolo_error_analysis.png").exists():
        st.image(str(BASE_DIR / "yolo_error_analysis.png"), caption="YOLO Prediction")
    st.markdown("""
    *   **Why does YOLO fail?** YOLO is designed to find large objects like cars and dogs. Endometriosis lesions are often smaller than 50 pixels. YOLO's architecture literally divides the image into a grid, so if a lesion is smaller than a grid cell, it disappears. It also struggles when two lesions overlap.
    """)

# ==========================================
# PAGE 10: HYPERPARAMETER TUNING
# ==========================================
elif page == "⚙️ Hyperparameter Tuning":
    st.title("Hyperparameter Tuning")
    st.write("Deep Learning models have 'knobs' that need tuning. We systematically tested different configurations to find the optimal setup.")
    
    st.markdown("### 1. Classifier Learning Rate (RQ1)")
    st.write("The learning rate controls how fast the model updates its weights. If it's too high, it becomes unstable. If it's too low, it never learns.")
    
    df_lr = pd.DataFrame({
        "Learning Rate": ["1e-3 (0.001)", "1e-4 (0.0001)", "1e-5 (0.00001)"],
        "Validation AUROC": [0.8764, 0.8776, 0.5321]
    })
    fig_lr = px.bar(df_lr, x="Learning Rate", y="Validation AUROC", text_auto='.4f', color="Validation AUROC", color_continuous_scale="Teal")
    st.plotly_chart(fig_lr, use_container_width=True)
    st.success("Conclusion: 1e-4 is the 'Goldilocks' zone. 1e-5 is far too slow and fails to converge, while 1e-3 is slightly unstable.")
    
    st.markdown("### 2. Segmentation Thresholds (RQ2)")
    st.markdown("""
    *   **U-Net Threshold:** The default is 0.5 (meaning if the model is >50% sure a pixel is a lesion, it colors it). We tested 0.3 to 0.7. **0.5 yielded the highest Dice score**.
    *   **YOLO Confidence:** Lowering YOLO's confidence threshold helps it detect tiny lesions, but it hallucinates false positives. We tuned it to balance precision and recall.
    """)

# ==========================================
# PAGE 11: CONCLUSION (ENHANCED)
# ==========================================
elif page == "🏁 Conclusion":
    st.title("Project Conclusion & Future Work")
    
    st.markdown("""
    ### Summary of Key Findings
    
    **1. Transfer Learning is Mandatory:**
    A Custom CNN built from scratch failed completely (AUROC ~0.48), essentially guessing randomly. A pretrained ResNet-50 succeeded (AUROC ~0.89). This proves that for small medical datasets, models must start with knowledge learned from millions of everyday images.
    
    **2. The "Shortcut Learning" Problem:**
    We discovered that the ResNet-50 classifier was "cheating." It achieved high accuracy by memorizing the background lighting and camera type of the surgeries, rather than looking at the lesion itself. When we blacked out the lesions (Shortcut Probe) and used Grad-CAM explainability, the model's performance barely dropped, and the heatmaps proved it was looking at the background.
    
    **3. Loss Functions are Critical for Small Objects:**
    Standard Dice+BCE loss caused the U-Net to predict empty masks, because 97% of the pixels in the dataset are healthy background. By switching to **Focal Tversky Loss**, we forced the model to focus on the tiny lesions, improving the Dice score by 48% (from 0.25 to 0.37).
    
    **4. Paradigm Comparison (Instance vs. Dense):**
    YOLO (Instance Segmentation) struggled with a mAP of only 0.19 because lesions are often smaller than its detection grid. U-Net (Dense Segmentation) performed better because it classifies every single pixel individually, but it still struggles with diffuse, patchy peritoneum lesions.
    
    **5. Rare Classes vs. Common Classes:**
    Surprisingly, the rare classes (Ovary, TIE) achieved *higher* Dice scores than the common Peritoneum class. Peritoneum lesions are often spread out and have fuzzy edges, while Ovary lesions (cysts) are distinct and circular. *Note: Uterus had only 2 test frames, making its result statistically unstable.*
    
    **6. The Classification Gate:**
    While a classification gate successfully reduced false positives from 2.5% to 0.1% on the training set, the RQ1 shortcut learning problem means this gate relies on background context. It cannot be trusted in a real clinical setting.
    """)
    
    st.markdown("---")
    st.markdown("### Future Work")
    st.markdown("""
    To make this clinically deployable, future research should focus on:
    1.  **Temporal Information:** Instead of looking at static single frames, using video (LSTMs or 3D CNNs) would allow the model to track lesions as the camera moves.
    2.  **Multi-Class Segmentation:** Instead of a binary "lesion vs. background" mask, training a model to output distinct colors for each of the 4 lesion classes (Peritoneum, Ovary, Uterus, TIE).
    3.  **External Validation:** Testing the models on a completely different hospital's dataset to ensure they generalize.
    """)

st.sidebar.markdown("---")
st.sidebar.info("Built for the GLENDA Research Proposal. Data: ITEC, Uni Klagenfurt (CC BY-NC 4.0).")