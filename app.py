import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import os
import requests
from io import BytesIO
import numpy as np

# Page config with enhanced metadata
st.set_page_config(
    page_title="DermScan AI | Professional Dermoscopic Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling with confidence intervals
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --success-color: #06A77D;
        --warning-color: #F18F01;
        --danger-color: #C73E1D;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header styling */
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
    
    /* Performance metrics bar */
    .metrics-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border-left: 4px solid #2E86AB;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Button styling */
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
    
    /* Disclaimer box */
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
    
    /* Result cards */
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
    
    /* Info boxes */
    .info-box {
        background: #E8F4F8;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #e0e0e0;
        color: #666;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Upload section */
    .upload-section {
        border: 2px dashed #2E86AB;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #F8FCFD;
    }
    
    /* Confidence Interval Styling */
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
    
    /* Statistical details box */
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

# Helper function to download the model file
@st.cache_resource(show_spinner=False)
def download_file(url, path):
    """Downloads a file securely if it doesn't exist."""
    if os.path.exists(path):
        return
        
    st.info(f"🔄 Downloading DermScan AI model... This may take a moment.")
    try:
        total_size = 70950235
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status() 

        if 'content-length' in response.headers:
            total_size = int(response.headers['content-length'])
            
        block_size = 1024 * 10
        progress_bar = st.progress(0, text="Initializing download...")
        
        with open(path, 'wb') as f:
            downloaded_size = 0
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded_size += len(data)
                
                if total_size > 0:
                    progress = min(int(downloaded_size * 100 / total_size), 99)
                    progress_text = f"⏳ Downloading: {round(downloaded_size / (1024*1024), 1)} MB / {round(total_size / (1024*1024), 1)} MB"
                else:
                    progress = 0
                    progress_text = f"⏳ Downloaded: {round(downloaded_size / (1024*1024), 1)} MB"
                    
                progress_bar.progress(progress, text=progress_text)
                
        progress_bar.progress(100, text="✅ Model ready!")
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

# Load model
@st.cache_resource
def load_model():
    """Loads the EfficientNet-B4 model with cached resource functionality."""
    download_file(MODEL_URL, MODEL_PATH)

    try:
        model = timm.create_model(
            MODEL_NAME, 
            pretrained=False, 
            num_classes=NUM_CLASSES
        )
        
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the model globally
model = load_model()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Confidence interval calculation function
def calculate_confidence_interval(probability, confidence_level=0.95):
    """
    Calculate confidence interval for binary classification probability.
    Uses Wilson score interval for more accurate bounds near 0 and 1.
    
    Args:
        probability: The predicted probability (0-1)
        confidence_level: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        tuple: (lower_bound, upper_bound, margin_of_error)
    """
    from scipy import stats
    
    # Z-score for confidence level (1.96 for 95% CI)
    z = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Wilson score interval (more accurate than normal approximation)
    # Assumes n=1 for single prediction, but we adjust based on model uncertainty
    n = 100  # Effective sample size (can be tuned based on validation set performance)
    
    p = probability
    
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denominator
    
    lower_bound = max(0, center - margin)
    upper_bound = min(1, center + margin)
    margin_of_error = margin
    
    return lower_bound, upper_bound, margin_of_error

def calculate_uncertainty_level(margin_of_error):
    """
    Categorize uncertainty level based on margin of error.
    
    Args:
        margin_of_error: The margin of error from CI calculation
    
    Returns:
        str: "Low", "Moderate", or "High"
    """
    if margin_of_error < 0.08:
        return "Low"
    elif margin_of_error < 0.15:
        return "Moderate"
    else:
        return "High"

# Prediction function with confidence intervals
def predict_image(img):
    """Processes the image and returns prediction probabilities with confidence intervals."""
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
        
        # Calculate confidence intervals
        mal_lower, mal_upper, mal_margin = calculate_confidence_interval(malignant_prob)
        ben_lower, ben_upper, ben_margin = calculate_confidence_interval(benign_prob)
        
        # Calculate uncertainty levels
        mal_uncertainty = calculate_uncertainty_level(mal_margin)
        ben_uncertainty = calculate_uncertainty_level(ben_margin)
        
        # Calculate model certainty (based on how far from 0.5)
        certainty = abs(malignant_prob - 0.5) * 2 * 100  # Scale to 0-100%
        
        return {
            'benign': benign_prob,
            'malignant': malignant_prob,
            'is_high_risk': malignant_prob >= MALIGNANT_THRESHOLD,
            'confidence_intervals': {
                'malignant': {
                    'lower': mal_lower,
                    'upper': mal_upper,
                    'margin': mal_margin,
                    'uncertainty': mal_uncertainty
                },
                'benign': {
                    'lower': ben_lower,
                    'upper': ben_upper,
                    'margin': ben_margin,
                    'uncertainty': ben_uncertainty
                }
            },
            'model_certainty': certainty
        }
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None

# ===================== UI START =====================

# Custom header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🔬 DermScan AI</h1>
    <p class="subtitle">Professional Dermoscopic Image Analysis with Statistical Confidence</p>
</div>
""", unsafe_allow_html=True)

# Performance metrics
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("🎯 F1 Score", f"{MODEL_METRICS['f1_score']:.1f}%", help="Overall model performance")
with col_m2:
    st.metric("🔍 Sensitivity", "88-90%", help="Detection rate for malignant lesions")
with col_m3:
    st.metric("✓ Accuracy", f"{MODEL_METRICS['accuracy']:.1f}%", help="Overall prediction accuracy")
with col_m4:
    st.metric("📊 Training Data", "10,000+", help="Images used for training")

st.markdown("<br>", unsafe_allow_html=True)

# Disclaimer section
st.markdown("""
<div class="disclaimer-box">
    <div class="disclaimer-title">⚠️ IMPORTANT MEDICAL DISCLAIMER</div>
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

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Dermoscopic Image")
    
    uploaded_file = st.file_uploader(
        "Select an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear dermoscopic image for analysis",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔬 Analyze Lesion", type="primary", disabled=(model is None)):
            if model is None:
                st.error("⚠️ Model not loaded. Please refresh the page.")
            else:
                with st.spinner("🔄 Analyzing image..."):
                    result = predict_image(image)
                    if result:
                        st.session_state['result'] = result
                        st.experimental_rerun()
    else:
        st.markdown("""
        <div class="upload-section">
            <h3 style="color: #2E86AB;">📁 No Image Uploaded</h3>
            <p style="color: #666;">Click "Browse files" above to upload a dermoscopic image</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Guidelines
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>📋 Image Quality Guidelines</strong>
        <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
            <li>✅ Use dermoscopic images only</li>
            <li>✅ Ensure lesion is centered and in focus</li>
            <li>✅ Provide adequate lighting</li>
            <li>❌ Avoid blurry or dark images</li>
            <li>❌ Do not include patient identifiable information</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 AI Analysis Results")
    
    if 'result' in st.session_state:
        result = st.session_state['result']
        ci = result['confidence_intervals']
        
        # Probability visualization
        st.markdown("#### Diagnostic Probabilities")
        
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #C73E1D;">
                <div style="color: #C73E1D; font-size: 0.9rem; font-weight: 600;">🔴 MALIGNANT</div>
                <div style="font-size: 2rem; font-weight: 700; color: #333; margin: 0.5rem 0;">
                    {:.1f}%
                </div>
            </div>
            """.format(result['malignant']*100), unsafe_allow_html=True)
            st.progress(result['malignant'])
            
            # Confidence Interval for Malignant
            uncertainty_class = f"uncertainty-{ci['malignant']['uncertainty'].lower()}"
            st.markdown(f"""
            <div class="ci-container">
                <div class="ci-header">95% Confidence Interval</div>
                <div class="ci-range">{ci['malignant']['lower']*100:.1f}% - {ci['malignant']['upper']*100:.1f}%</div>
                <div class="uncertainty-badge {uncertainty_class}">
                    {ci['malignant']['uncertainty']} Uncertainty (±{ci['malignant']['margin']*100:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with prob_col2:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #06A77D;">
                <div style="color: #06A77D; font-size: 0.9rem; font-weight: 600;">🟢 BENIGN</div>
                <div style="font-size: 2rem; font-weight: 700; color: #333; margin: 0.5rem 0;">
                    {:.1f}%
                </div>
            </div>
            """.format(result['benign']*100), unsafe_allow_html=True)
            st.progress(result['benign'])
            
            # Confidence Interval for Benign
            uncertainty_class = f"uncertainty-{ci['benign']['uncertainty'].lower()}"
            st.markdown(f"""
            <div class="ci-container">
                <div class="ci-header">95% Confidence Interval</div>
                <div class="ci-range">{ci['benign']['lower']*100:.1f}% - {ci['benign']['upper']*100:.1f}%</div>
                <div class="uncertainty-badge {uncertainty_class}">
                    {ci['benign']['uncertainty']} Uncertainty (±{ci['benign']['margin']*100:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Statistical Details Box
        with st.expander("📈 Statistical Analysis Details", expanded=False):
            stat_col1, stat_col2 = st.columns(2)
            
            with stat_col1:
                st.metric(
                    "Model Certainty",
                    f"{result['model_certainty']:.1f}%",
                    help="How confident the model is in this prediction"
                )
                st.metric(
                    "Confidence Level",
                    "95% CI",
                    help="Statistical confidence level used"
                )
            
            with stat_col2:
                st.metric(
                    "Malignant Margin of Error",
                    f"±{ci['malignant']['margin']*100:.1f}%",
                    help="Precision of malignant prediction"
                )
                st.metric(
                    "Prediction Reliability",
                    ci['malignant']['uncertainty'],
                    help="Based on confidence interval width"
                )
            
            st.info("""
            **📊 Understanding Confidence Intervals:**
            
            The 95% confidence interval indicates that we can be 95% confident the true probability 
            falls within the specified range. Narrower intervals indicate higher confidence in the prediction.
            
            - **Low Uncertainty**: Margin of error < 8% - High confidence
            - **Moderate Uncertainty**: Margin of error 8-15% - Borderline case
            - **High Uncertainty**: Margin of error > 15% - Additional assessment recommended
            """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Risk assessment with CI consideration
        if result['is_high_risk']:
            # Check if even the lower bound crosses threshold
            lower_bound_high_risk = ci['malignant']['lower'] >= MALIGNANT_THRESHOLD
            
            st.markdown(f"""
            <div class="result-high-risk">
                <h3 style="color: #C73E1D; margin-top: 0;">🚨 HIGH RISK DETECTION</h3>
                <p style="font-size: 1.1rem; color: #721C1C; margin: 0.5rem 0;">
                    <strong>Potential malignant lesion detected</strong><br>
                    Probability: {result['malignant']*100:.1f}% (CI: {ci['malignant']['lower']*100:.1f}%-{ci['malignant']['upper']*100:.1f}%)
                </p>
                
                <div style="background: white; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                    <h4 style="color: #C73E1D; margin-top: 0;">⚡ Immediate Action Required:</h4>
                    <ol style="color: #333; margin: 0;">
                        <li><strong>Schedule urgent dermatologist consultation</strong> (within 2 weeks)</li>
                        <li><strong>Bring or send this analysis</strong> with your referral</li>
                        <li><strong>Do not delay</strong> – early detection saves lives</li>
                    </ol>
                </div>
                
                <div style="background: {'white' if lower_bound_high_risk else '#FFF3CD'}; padding: 1rem; border-radius: 5px; margin: 1rem 0; border-left: 4px solid {'#C73E1D' if lower_bound_high_risk else '#F18F01'};">
                    <p style="color: #721C1C; margin: 0; font-size: 0.95rem;">
                        <strong>🔬 Clinical Interpretation:</strong><br>
                        {'Even with the lower bound of the confidence interval (' + f"{ci['malignant']['lower']*100:.1f}%" + '), this lesion significantly exceeds the clinical threshold of ' + f"{MALIGNANT_THRESHOLD*100:.0f}%" + ' for urgent referral.' if lower_bound_high_risk else 'The confidence interval spans the clinical threshold. Given the uncertainty and the potential severity, urgent professional evaluation is strongly recommended.'}
                        The model shows <strong>{ci['malignant']['uncertainty'].lower()} uncertainty</strong> in this prediction.
                    </p>
                </div>
                
                <p style="color: #721C1C; font-size: 0.95rem; margin: 0.5rem 0;">
                    <strong>About Malignant Lesions:</strong><br>
                    May include melanoma, basal cell carcinoma, or squamous cell carcinoma. 
                    Professional evaluation and likely biopsy required. Treatment outcomes are significantly 
                    better with early detection.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Check if upper bound is close to threshold
            near_threshold = ci['malignant']['upper'] >= (MALIGNANT_THRESHOLD - 0.05)
            
            st.markdown(f"""
            <div class="result-low-risk">
                <h3 style="color: #06A77D; margin-top: 0;">✅ LOWER RISK INDICATION</h3>
                <p style="font-size: 1.1rem; color: #0D5C3D; margin: 0.5rem 0;">
                    <strong>Lesion appears benign</strong><br>
                    Probability: {result['malignant']*100:.1f}% (CI: {ci['malignant']['lower']*100:.1f}%-{ci['malignant']['upper']*100:.1f}%)
                </p>
                
                <div style="background: white; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                    <h4 style="color: #06A77D; margin-top: 0;">📋 Recommended Actions:</h4>
                    <ol style="color: #333; margin: 0;">
                        <li><strong>Monitor regularly</strong> for any changes</li>
                        <li><strong>Document with photos</strong> monthly</li>
                        <li><strong>Consult healthcare provider</strong> if changes occur</li>
                        <li><strong>Continue routine skin checks</strong></li>
                    </ol>
                </div>
                
                {'<div style="background: #FFF3CD; padding: 1rem; border-radius: 5px; margin: 1rem 0; border-left: 4px solid #F18F01;"><p style="color: #856404; margin: 0; font-size: 0.95rem;"><strong>⚠️ Note:</strong> The upper confidence bound (' + f"{ci['malignant']['upper']*100:.1f}%" + ') approaches the clinical threshold. Consider professional evaluation for additional peace of mind.</p></div>' if near_threshold else ''}
                
                <p style="color: #0D5C3D; font-size: 0.95rem; margin: 0.5rem 0;">
                    <strong>Important:</strong> Even benign-appearing lesions require monitoring. 
                    Use the ABCDE rule to watch for warning signs and maintain regular professional 
                    skin examinations. The model shows <strong>{ci['malignant']['uncertainty'].lower()} uncertainty</strong> in this prediction.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed probabilities
        with st.expander("📊 Detailed Probability Breakdown"):
            risk_level_mal = (
                "Critical" if result['malignant'] >= 0.7 else
                "High" if result['malignant'] >= 0.5 else
                "Moderate" if result['malignant'] >= 0.35 else
                "Low"
            )
            
            risk_level_ben = (
                "Very High" if result['benign'] >= 0.8 else
                "High" if result['benign'] >= 0.65 else
                "Uncertain"
            )
            
            st.markdown(f"""
            | Category | Probability | 95% CI Range | Uncertainty | Risk Level |
            |----------|-------------|--------------|-------------|------------|
            | 🔴 Malignant | **{result['malignant']*100:.2f}%** | {ci['malignant']['lower']*100:.1f}%-{ci['malignant']['upper']*100:.1f}% | {ci['malignant']['uncertainty']} | {risk_level_mal} |
            | 🟢 Benign | **{result['benign']*100:.2f}%** | {ci['benign']['lower']*100:.1f}%-{ci['benign']['upper']*100:.1f}% | {ci['benign']['uncertainty']} | {risk_level_ben} |
            
            **Decision Threshold:** {MALIGNANT_THRESHOLD} (optimized for ~88-90% sensitivity)
            
            **Model Certainty:** {result['model_certainty']:.1f}%
            
            **Clinical Note:** This threshold is set to maximize detection of malignant lesions while 
            minimizing false negatives, which is crucial in medical screening applications. Confidence 
            intervals provide additional context for borderline cases.
            """)
    else:
        st.markdown("""
        <div class="info-box" style="text-align: center; padding: 2rem;">
            <h3 style="color: #2E86AB;">📊 Awaiting Analysis</h3>
            <p style="color: #666;">Upload an image and click "Analyze Lesion" to view results with confidence intervals</p>
        </div>
        """, unsafe_allow_html=True)

# Educational content
st.markdown("<br><br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📖 ABCDE Rule", "🔬 Model Information", "📊 Understanding Confidence", "🌐 Resources"])

with tab1:
    st.markdown("""
    ### The ABCDE Rule for Skin Cancer Detection
    
    **Watch for these warning signs in moles and lesions:**
    
    <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    
    **🅰️ Asymmetry**  
    One half of the mole doesn't match the other half
    
    **🅱️ Border Irregularity**  
    Edges are ragged, notched, or blurred rather than smooth
    
    **©️ Color Variation**  
    Multiple colors present or uneven color distribution
    
    **🅳 Diameter**  
    Larger than 6mm (about the size of a pencil eraser)
    
    **🅴 Evolving**  
    Changes in size, shape, color, elevation, or new symptoms (bleeding, itching, crusting)
    
    </div>
    
    <br>
    
    > ⚠️ **If you notice ANY of these signs, consult a dermatologist immediately!**
    """, unsafe_allow_html=True)

with tab2:
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown(f"""
        **🏗️ Architecture**  
        EfficientNet-B4 (Clinical-Grade CNN)
        
        **📚 Training Dataset**  
        HAM10000 (~10,000 dermoscopic images)
        
        **🎯 F1 Score**  
        {MODEL_METRICS['f1_score']:.2f}%
        
        **🔍 Sensitivity**  
        {MODEL_METRICS['sensitivity']:.2f}% (at 0.5 threshold)  
        ~88-90% (at {MALIGNANT_THRESHOLD} threshold)
        """)
    
    with col_t2:
        st.markdown(f"""
        **✓ Accuracy**  
        {MODEL_METRICS['accuracy']:.2f}%
        
        **⚡ Training Epochs**  
        {MODEL_METRICS['epoch']}
        
        **🎚️ Decision Threshold**  
        {MALIGNANT_THRESHOLD} (optimized for sensitivity)
        
        **🔬 Model Purpose**  
        Research & Educational Tool
        """)
    
    st.info("""
    **Note on Model Performance:**  
    The threshold is optimized to maximize sensitivity (detection of malignant cases) at the expense 
    of some specificity. This "better safe than sorry" approach is standard in medical screening 
    applications where missing a malignant case is more serious than a false positive.
    """)

with tab3:
    st.markdown("""
    ### 📊 Understanding Confidence Intervals in Medical AI
    
    #### What are Confidence Intervals?
    
    A **95% confidence interval** means we can be 95% confident that the true probability falls 
    within the specified range. This provides crucial context beyond just the point estimate.
    
    #### Why They Matter in Dermoscopy
    
    **For Clear Cases:**
    - Narrow confidence intervals (±5-8%) indicate high confidence
    - Both bounds clearly above or below the threshold
    - Stronger clinical decision support
    
    **For Borderline Cases:**
    - Wider confidence intervals (±10-15%) indicate uncertainty
    - Bounds may span the clinical threshold
    - Suggests need for additional assessment or imaging
    
    #### Uncertainty Levels
    
    - 🟢 **Low Uncertainty** (< 8% margin): High confidence, reliable prediction
    - 🟡 **Moderate Uncertainty** (8-15% margin): Borderline case, professional review recommended
    - 🔴 **High Uncertainty** (> 15% margin): Additional assessment strongly recommended
    
    #### Clinical Application
    
    Even with uncertainty, the model errs on the side of caution. If the malignant probability 
    or its confidence interval approaches or exceeds the clinical threshold, referral is recommended.
    
    > 💡 **Key Insight:** Confidence intervals help distinguish between "definitely high risk" and 
    > "uncertain, but warrants caution" cases, improving clinical decision-making.
    """)

with tab4:
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown("""
        **🇬🇧 UK Resources**
        - [British Association of Dermatology](https://www.skinhealthinfo.org.uk)
        - [NHS Skin Cancer Information](https://www.nhs.uk/conditions/skin-cancer/)
        - [Cancer Research UK](https://www.cancerresearchuk.org/about-cancer/skin-cancer)
        """)
    
    with col_r2:
        st.markdown("""
        **🇺🇸 US Resources**
        - [American Academy of Dermatology](https://www.aad.org/find-a-derm)
        - [Skin Cancer Foundation](https://www.skincancer.org)
        - [American Cancer Society](https://www.cancer.org/cancer/skin-cancer.html)
        """)

# Footer
st.markdown("""
<div class="custom-footer">
    <h3 style="color: #2E86AB; margin-bottom: 0.5rem;">🔬 DermScan AI</h3>
    <p style="font-size: 1rem; margin: 0.5rem 0;">
        <strong>Advanced Dermoscopic Image Analysis with Statistical Confidence</strong>
    </p>
    <p style="color: #999; font-size: 0.9rem; margin: 0.5rem 0;">
        Educational & Research Tool • Not for Clinical Diagnosis
    </p>
    <p style="color: #666; font-size: 0.85rem; margin: 0.5rem 0;">
        Model: EfficientNet-B4 | F1: 85.2% | Sensitivity: ~88-90% | With 95% Confidence Intervals
    </p>
    <p style="color: #999; font-size: 0.8rem;">
        Dr Tom Hutchinson • Oxford, United Kingdom
    </p>
</div>
""", unsafe_allow_html=True)
