import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import os
import requests
from io import BytesIO
import numpy as np
from scipy import stats

# Page config
st.set_page_config(
    page_title="DermScan AI | Professional Dermoscopic Analysis",
    page_icon="microscope",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (unchanged)
st.markdown("""
<style>
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --success-color: #06A77D;
        --warning-color: #F18F01;
        --danger-color: #C73E1D;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main-header {
        background: linear-gradient(135deg, #2E86AB 0%, #1a4d6d 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    .subtitle {
        color: #E8F4F8;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .metrics-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border-left: 4px solid #2E86AB;
    }
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2E86AB 0%, #1a4d6d 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.4);
    }
    .disclaimer-box {
        background: #FFF3CD;
        border: 2px solid #F18F01;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .disclaimer-title {
        color: #856404;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-high-risk {
        background: linear-gradient(135deg, #FFF5F5 0%, #FFE5E5 100%);
        border-left: 5px solid #C73E1D;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-low-risk {
        background: linear-gradient(135deg, #F0FFF4 0%, #E5F9E7 100%);
        border-left: 5px solid #06A77D;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background: #E8F4F8;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .custom-footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #e0e0e0;
        color: #666;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    .upload-section {
        border: 2px dashed #2E86AB;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #F8FCFD;
    }
    .ci-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.75rem;
        border: 1px solid #e0e0e0;
    }
    .ci-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .ci-range {
        font-size: 0.95rem;
        color: #333;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .uncertainty-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .uncertainty-low {
        background: #E5F9E7;
        color: #06A77D;
        border: 1px solid #06A77D;
    }
    .uncertainty-moderate {
        background: #FFF3CD;
        color: #F18F01;
        border: 1px solid #F18F01;
    }
    .uncertainty-high {
        background: #FFE5E5;
        color: #C73E1D;
        border: 1px solid #C73E1D;
    }
    .stats-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e8e8e8;
        margin-top: 1rem;
    }
    .stats-box h4 {
        color: #2E86AB;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MODEL_NAME = "efficientnet_b4"
IMG_SIZE = 512
NUM_CLASSES = 2
CLASS_NAMES = ["Benign", "Malignant"]
MALIGNANT_THRESHOLD = 0.35
MODEL_URL = "https://huggingface.co/Skindoc/streamlitapp/resolve/main/model.pth"
MODEL_PATH = "model_cache.pth"
MODEL_METRICS = {
    'f1_score': 85.24,
    'sensitivity': 83.01,
    'accuracy': 88.47,
    'epoch': 40
}

# Download model
@st.cache_resource(show_spinner=False)
def download_file(url, path):
    if os.path.exists(path):
        return
    st.info("Downloading DermScan AI model... This may take a moment.")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 10
        progress_bar = st.progress(0, text="Initializing...")
        downloaded = 0
        with open(path, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)
                if total_size > 0:
                    progress = min(int(downloaded * 100 / total_size), 99)
                    text = f"Downloading: {downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB"
                else:
                    progress = 0
                    text = f"Downloaded: {downloaded/(1024*1024):.1f} MB"
                progress_bar.progress(progress, text=text)
        progress_bar.progress(100, text="Model ready!")
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()

# Load model with warmup
@st.cache_resource
def load_model():
    download_file(MODEL_URL, MODEL_PATH)
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()

        # Warmup
        with st.spinner("Warming up AI model..."):
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            with torch.no_grad():
                _ = model(dummy)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model globally
model = load_model()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Confidence interval
def calculate_confidence_interval(probability, confidence_level=0.95):
    z = stats.norm.ppf((1 + confidence_level) / 2)
    n = 100
    p = probability
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denominator
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    return lower, upper, margin

def calculate_uncertainty_level(margin):
    if margin < 0.08:
        return "Low"
    elif margin < 0.15:
        return "Moderate"
    else:
        return "High"

# Prediction
def predict_image(img):
    if img is None or model is None:
        return None
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
        benign_prob = float(probs[0])
        malignant_prob = float(probs[1])
        mal_lower, mal_upper, mal_margin = calculate_confidence_interval(malignant_prob)
        ben_lower, ben_upper, ben_margin = calculate_confidence_interval(benign_prob)
        mal_uncertainty = calculate_uncertainty_level(mal_margin)
        ben_uncertainty = calculate_uncertainty_level(ben_margin)
        certainty = abs(malignant_prob - 0.5) * 200
        return {
            'benign': benign_prob,
            'malignant': malignant_prob,
            'is_high_risk': malignant_prob >= MALIGNANT_THRESHOLD,
            'confidence_intervals': {
                'malignant': {'lower': mal_lower, 'upper': mal_upper, 'margin': mal_margin, 'uncertainty': mal_uncertainty},
                'benign': {'lower': ben_lower, 'upper': ben_upper, 'margin': ben_margin, 'uncertainty': ben_uncertainty}
            },
            'model_certainty': certainty
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ===================== UI =====================
st.markdown("""
<div class="main-header">
    <h1 class="main-title">DermScan AI</h1>
    <p class="subtitle">Professional Dermoscopic Image Analysis with Statistical Confidence</p>
</div>
""", unsafe_allow_html=True)

# Metrics
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("F1 Score", f"{MODEL_METRICS['f1_score']:.1f}%", help="Overall model performance")
with col_m2:
    st.metric("Sensitivity", "~88-90%", help="At 0.35 threshold (optimized for screening)")
with col_m3:
    st.metric("Accuracy", f"{MODEL_METRICS['accuracy']:.1f}%", help="Overall prediction accuracy")
with col_m4:
    st.metric("Training Data", "10,000+", help="Images used for training")
st.markdown("<br>", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <div class="disclaimer-title">IMPORTANT MEDICAL DISCLAIMER</div>
    <p style="margin: 0.5rem 0; color: #856404; font-size: 1rem;">
        <strong>FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY</strong>
    </p>
    <ul style="color: #856404; margin: 0.5rem 0; padding-left: 1.5rem;">
        <li><strong>NOT FDA/NICE APPROVED</strong> - Not intended for diagnostic use</li>
        <li><strong>ALWAYS</strong> consult a qualified dermatologist or medical professional</li>
        <li>Only a <strong>biopsy</strong> can definitively diagnose skin cancer</li>
        <li>This tool is for educational research purposes only</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Upload Dermoscopic Image")
    uploaded_file = st.file_uploader(
        "Select an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear dermoscopic image for analysis",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Clear previous results on new upload
        for key in ['result', 'analyzing']:
            if key in st.session_state:
                del st.session_state[key]

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Analyze Lesion", type="primary", disabled=(model is None)):
            if model is None:
                st.error("Model not loaded. Please refresh.")
            else:
                if st.session_state.get('analyzing', False):
                    st.warning("Analysis already in progress...")
                else:
                    st.session_state.analyzing = True
                    with st.spinner("Analyzing dermoscopic image..."):
                        try:
                            result = predict_image(image)
                            if result:
                                st.session_state['result'] = result
                                st.session_state.analyzing = False
                                st.success("Analysis complete!")
                            else:
                                st.session_state.analyzing = False
                                st.error("No result. Try another image.")
                        except Exception as e:
                            st.session_state.analyzing = False
                            st.error(f"Error: {e}")
                    st.rerun()
    else:
        st.markdown("""
        <div class="upload-section">
            <h3 style="color: #2E86AB;">No Image Uploaded</h3>
            <p style="color: #666;">Click "Browse files" to upload a dermoscopic image</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Image Quality Guidelines</strong>
        <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
            <li>Use dermoscopic images only</li>
            <li>Ensure lesion is centered and in focus</li>
            <li>Provide adequate lighting</li>
            <li>Avoid blurry or dark images</li>
            <li>Do not include patient identifiable information</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### AI Analysis Results")
    if 'result' in st.session_state:
        result = st.session_state['result']
        ci = result['confidence_intervals']
        # [Rest of results UI — unchanged from your original]
        # ... (same as your original code)
        # For brevity, I'll omit the full results block — it's unchanged
        # Paste your original results rendering code here
    else:
        st.markdown("""
        <div class="info-box" style="text-align: center; padding: 2rem;">
            <h3 style="color: #2E86AB;">Awaiting Analysis</h3>
            <p style="color: #666;">Upload an image and click "Analyze Lesion" to view results</p>
        </div>
        """, unsafe_allow_html=True)

# [Keep your tabs and footer unchanged below]
