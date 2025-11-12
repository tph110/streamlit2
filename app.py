import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import os
import requests
from io import BytesIO

# Page config
st.set_page_config(
    page_title="DermScan - Dermoscope image analysis",
    page_icon="🩺",
    layout="wide"
)

# Configuration
MODEL_NAME = "efficientnet_b4"
IMG_SIZE = 512
NUM_CLASSES = 2
CLASS_NAMES = ["Benign", "Malignant"]
MALIGNANT_THRESHOLD = 0.35

# FINAL FIX: Using the direct Hugging Face URL to bypass Git LFS issues.
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
        
    st.info(f"Downloading DermScan model engine... This might take a moment.")
    try:
        # Initial estimate for total size
        # Using the actual size for a more accurate progress bar
        total_size = 70950235
        
        response = requests.get(url, stream=True, timeout=60) # Increased timeout for large files
        response.raise_for_status() 

        # Use actual content-length if available
        if 'content-length' in response.headers:
            total_size = int(response.headers['content-length'])
            
        block_size = 1024 * 10 # 10 Kibibytes for faster progress updates
        
        progress_bar = st.progress(0, text="Download progress...")
        
        with open(path, 'wb') as f:
            downloaded_size = 0
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded_size += len(data)
                
                # Ensure we don't divide by zero
                if total_size > 0:
                    progress = min(int(downloaded_size * 100 / total_size), 99)
                    progress_text = f"Download progress: {round(downloaded_size / (1024*1024), 1)} MB of {round(total_size / (1024*1024), 1)} MB"
                else:
                    progress = 0
                    progress_text = f"Download progress: {round(downloaded_size / (1024*1024), 1)} MB downloaded"
                    
                progress_bar.progress(progress, text=progress_text)
                
        progress_bar.progress(100, text="Model download complete!")
        st.success("Model ready.")
    except Exception as e:
        st.error(f"Failed to download model from {url}. Error: {e}")
        st.stop() # Stop execution if model download fails

# Load model
@st.cache_resource
def load_model():
    """Loads the EfficientNet-B4 model with cached resource functionality."""
    
    # 1. Download the file first
    download_file(MODEL_URL, MODEL_PATH)

    try:
        model = timm.create_model(
            MODEL_NAME, 
            pretrained=False, 
            num_classes=NUM_CLASSES
        )
        
        # 2. Load the model from the local downloaded path
        # weights_only=False resolves the PyTorch 2.6+ compatibility issue
        # Note: We are now loading the correct binary file, so the 'invalid load key' error should be fixed here.
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. **Action Required:** Please ensure your MODEL_URL is correct and the downloaded file is a complete PyTorch binary.")
        return None

# Load the model globally
model = load_model()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # Standard normalization for ImageNet-trained models
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(img):
    """Processes the image and returns prediction probabilities."""
    # Check if the model loaded successfully (model will be None if it failed)
    if img is None or model is None:
        # The 'NoneType' object is not callable error is solved by fixing the model loading above.
        return None
    
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Apply preprocessing and add batch dimension
        x = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(x)
            # Apply softmax to convert logits to probabilities
            probs = torch.softmax(logits, dim=1)[0]
        
        # Assuming index 0 is Benign and index 1 is Malignant
        benign_prob = float(probs[0])
        malignant_prob = float(probs[1])
        
        return {
            'benign': benign_prob,
            'malignant': malignant_prob,
            'is_high_risk': malignant_prob >= MALIGNANT_THRESHOLD
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# UI
st.title("🩺 DermScan - Dermoscopic Imaging Analysis Tool")
st.markdown(f"**Clinical Performance:** F1: {MODEL_METRICS['f1_score']:.1f}% | Sensitivity: ~88-90% | Accuracy: {MODEL_METRICS['accuracy']:.1f}% | Trained on over 10,000 images")

# Warning box
st.error("""
**⚠️ DISCLAIMER**

**FOR RESEARCH PURPOSES ONLY**

- 🚫 Not to be used for diagnostic purposes
- ✅ ALWAYS consult a trained medical professional
- 🔬 Only a biopsy can definitively diagnose or exclude skin cancer
- ⚖️ Not FDA/NICE approved • For educational and research purposes only
""")

# Two columns layout for upload and results
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Dermoscope Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear dermascope photo"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Button to trigger analysis
        if st.button("🔍 Analyse Lesion", type="primary", use_container_width=True, disabled=(model is None)):
            if model is None:
                st.warning("Cannot analyze: Model failed to load. Please fix the 'Error loading model' issue above.")
            else:
                with st.spinner("Analyzing..."):
                    result = predict_image(image)
                    
                    if result:
                        st.session_state['result'] = result
    
    st.info("""
    **📋 Image Guidelines:**
    - ✅ Dermatoscope images only
    - ✅ Ensure lesion is centered and in focus
    - ✅ Ensure optimal lighting
    - ⚠️ Avoid blurry or dark images
    - ⚠️ Do not upload any patient identifiable data 
    """)

with col2:
    st.subheader("📊 AI Analysis Results")
    
    if 'result' in st.session_state:
        result = st.session_state['result']
        
        # Progress bars
        st.markdown("### Probability Scores")
        
        mal_col, ben_col = st.columns(2)
        
        with mal_col:
            st.metric(
                "🔴 Malignant Risk",
                f"{result['malignant']*100:.1f}%",
                delta=None
            )
            # Use a custom color for malignant progress bar if possible (Streamlit default)
            st.progress(result['malignant'])
        
        with ben_col:
            st.metric(
                "🟢 Benign",
                f"{result['benign']*100:.1f}%",
                delta=None
            )
            st.progress(result['benign'])
        
        st.markdown("---")
        
        # Risk assessment
        if result['is_high_risk']:
            st.error(f"""
            ### 🚨 HIGH RISK - Potential Malignant Lesion Detected (P > {MALIGNANT_THRESHOLD})
            
            **Immediate Actions Required:**
            1. 📅 Warrants an urgent referral to a Dermatologist within 2 weeks
            2. 📸 Include this image in the referral
            3. ⏰ Do not delay - early detection is critical
            
            **About Malignant Lesions:**
            - Can include melanoma, basal cell carcinoma, or squamous cell carcinoma
            - Requires professional evaluation and likely biopsy
            - Treatment success is highest with early detection
            """)
        else:
            st.success(f"""
            ### ✅ LOWER RISK - Appears Benign (P < {MALIGNANT_THRESHOLD})
            
            **Recommended Actions:**
            1. 👁️ Monitor the lesion regularly for changes
            2. 📸 Take monthly photos to track changes
            3. 🏥 To be reviewed again by a healthcare professional immediately if any new changes occur
            
            **Remember:**
            - Even benign lesions should be monitored
            - Use the ABCDE rule to watch for warning signs
            - Regular skin checks are essential
            """)
        
        st.markdown("---")
        
        # Detailed probabilities
        with st.expander("📊 Detailed Probability Breakdown"):
            st.markdown(f"""
            | Category | Probability | Interpretation |
            |----------|-------------|----------------|
            | 🔴 Malignant | {result['malignant']*100:.1f}% | {'High risk' if result['malignant'] >= 0.7 else 'Moderate risk' if result['malignant'] >= 0.5 else 'Low-moderate risk' if result['malignant'] >= 0.35 else 'Lower risk'} |
            | 🟢 Benign | {result['benign']*100:.1f}% | {'Very likely benign' if result['benign'] >= 0.8 else 'Probably benign' if result['benign'] >= 0.65 else 'Uncertain'} |
            
            **Decision Threshold:** {MALIGNANT_THRESHOLD} (optimized to catch ~88-90% of malignant cases)
            """)
    else:
        st.info("Upload an image and click 'Analyse Lesion' to see results")

# Expandable sections
with st.expander("📖 The ABCDE Rule for Monitoring"):
    st.markdown("""
    Watch for these warning signs:
    
    - **A - Asymmetry:** One half doesn't match the other
    - **B - Border:** Irregular, ragged, or blurred edges
    - **C - Color:** Multiple colors or uneven distribution
    - **D - Diameter:** Larger than 6mm (pencil eraser)
    - **E - Evolving:** Changes in size, shape, or color
    
    **If you notice ANY of these, see a dermatologist immediately!** """)

with st.expander("🔬 About This Model"):
    st.markdown(f"""
    **Model Details:**
    - Architecture: EfficientNet-B4 (Clinical-Grade)
    - Training: HAM10000 dataset (~10,000 images)
    - F1 Score: {MODEL_METRICS['f1_score']:.2f}%
    - Sensitivity: {MODEL_METRICS['sensitivity']:.2f}% (at 0.5 threshold)
    - Enhanced Sensitivity: ~88-90% (at {MALIGNANT_THRESHOLD} threshold)
    - Accuracy: {MODEL_METRICS['accuracy']:.2f}%
    
    """)

with st.expander("🌐 Additional Resources"):
    st.markdown("""
    **Find Professional Help:**
    - [British Association of Dermatology] (https://www.skinhealthinfo.org.uk)
    - [American Academy of Dermatology - Find a Dermatologist](https://www.aad.org/find-a-derm)
    - [Skin Cancer Foundation](https://www.skincancer.org)
    - [American Cancer Society - Skin Cancer](https://www.cancer.org/cancer/skin-cancer.html)
    
    """)

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #666;">
<strong>🩺 DermScan - Dermoscopic image analysis tool</strong><br>
<em>Educational tool • Not for medical diagnosis</em><br>
<small>Model: EfficientNet-B4 | F1: 85.2% | Dr Tom Hutchinson, Oxford, UK</small>
</p>
""", unsafe_allow_html=True)
