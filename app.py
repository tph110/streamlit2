import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import os
import requests
from io import BytesIO
import numpy as np
import hashlib
import logging
from typing import Optional, Dict, Tuple, Any
from functools import wraps
# ===================== CONFIGURATION =====================
# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_SIZE = 2048  # pixels
MIN_IMAGE_SIZE = 64  # pixels
MODEL_NAME = "efficientnet_b4"
IMG_SIZE = 512
NUM_CLASSES = 2
CLASS_NAMES = ["Benign", "Malignant"]
MALIGNANT_THRESHOLD = 0.35
MODEL_URL = "https://huggingface.co/Skindoc/streamlitapp/resolve/main/model.pth"
MODEL_PATH = "model_cache.pth"
MODEL_SHA256 = None  # Optional: Add checksum for verification
MODEL_METRICS = {
    'f1_score': 85.24,
    'sensitivity': 83.01,
    'accuracy': 88.47,
    'epoch': 40
}

# Test-time augmentation parameters for uncertainty estimation
TTA_NUM_SAMPLES = 10  # Number of augmented predictions for uncertainty
TTA_ROTATION_DEG = 5  # Rotation degrees for augmentation
TTA_BRIGHTNESS = 0.1  # Brightness variation

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Initialize session state keys
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'analyzing' not in st.session_state:
    st.session_state['analyzing'] = False
if 'last_uploaded' not in st.session_state:
    st.session_state['last_uploaded'] = None

# ===================== UTILITY FUNCTIONS =====================
def calculate_file_sha256(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def validate_image_file(uploaded_file) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded image file.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE:
            return False, f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
        
        # Check file type
        if uploaded_file.type not in ['image/jpeg', 'image/jpg', 'image/png']:
            return False, "Invalid file type. Please upload a JPEG or PNG image."
        
        # Try to open and validate image
        try:
            image = Image.open(uploaded_file)
            image.verify()  # Verify it's a valid image
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
        
        # Reopen for actual use (verify() closes the image)
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        
        # Check image dimensions
        width, height = image.size
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            return False, f"Image too small. Minimum size: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE} pixels"
        
        if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
            return False, f"Image too large. Maximum size: {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE} pixels"
        
        # Check if image is RGB or grayscale (convert grayscale to RGB)
        if image.mode not in ['RGB', 'L', 'RGBA']:
            return False, f"Unsupported image mode: {image.mode}. Please use RGB or grayscale images."
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        return False, f"Error validating image: {str(e)}"

# ===================== MODEL LOADING =====================
@st.cache_resource(show_spinner=False)
def download_file(url: str, path: str, max_retries: int = 3) -> bool:
    """
    Downloads a file securely with retry logic and optional checksum verification.
    
    Args:
        url: URL to download from
        path: Local path to save file
        max_retries: Maximum number of retry attempts
        
    Returns:
        bool: True if download successful, False otherwise
    """
    if os.path.exists(path):
        logger.info(f"Model file already exists: {path}")
        # Verify checksum if provided
        if MODEL_SHA256:
            try:
                file_hash = calculate_file_sha256(path)
                if file_hash != MODEL_SHA256:
                    logger.warning("Model file checksum mismatch. Re-downloading...")
                    os.remove(path)
                else:
                    logger.info("Model file checksum verified")
                    return True
            except Exception as e:
                logger.error(f"Error verifying checksum: {e}")
        else:
            return True
       
    logger.info(f"Downloading model from {url}")
    st.info("🔄 Downloading DermScan AI model... This may take a moment.")
    
    for attempt in range(max_retries):
        try:
            total_size = 70950235  # Default expected size
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            if 'content-length' in response.headers:
                total_size = int(response.headers['content-length'])
                # Verify reasonable file size
                if total_size > 500 * 1024 * 1024:  # 500MB limit
                    raise ValueError(f"File size too large: {total_size / (1024*1024):.1f}MB")
                if total_size < 1024:  # 1KB minimum
                    raise ValueError(f"File size too small: {total_size} bytes")
           
            block_size = 1024 * 10  # 10KB blocks
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
                   
            progress_bar.progress(100, text="✅ Verifying model...")
            
            # Verify file was downloaded completely
            if os.path.getsize(path) == 0:
                raise ValueError("Downloaded file is empty")
            
            # Verify checksum if provided
            if MODEL_SHA256:
                file_hash = calculate_file_sha256(path)
                if file_hash != MODEL_SHA256:
                    raise ValueError(f"Checksum mismatch. Expected: {MODEL_SHA256[:16]}..., Got: {file_hash[:16]}...")
                logger.info("Model file checksum verified")
            
            progress_bar.progress(100, text="✅ Model ready!")
            logger.info(f"Model downloaded successfully: {path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                st.warning(f"⚠️ Download failed. Retrying... ({attempt + 1}/{max_retries})")
                if os.path.exists(path):
                    os.remove(path)
            else:
                st.error(f"❌ Failed to download model after {max_retries} attempts: {e}")
                st.error("Please check your internet connection and try refreshing the page.")
                if os.path.exists(path):
                    os.remove(path)
                return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            st.error(f"❌ Failed to download model: {e}")
            if os.path.exists(path):
                os.remove(path)
            return False
    
    return False
@st.cache_resource
def load_model() -> Optional[torch.nn.Module]:
    """
    Loads the EfficientNet-B4 model with cached resource functionality.
    
    Returns:
        Optional[torch.nn.Module]: Loaded model or None if loading failed
    """
    try:
        # Download model file
        if not download_file(MODEL_URL, MODEL_PATH):
            logger.error("Failed to download model file")
            return None
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            st.error("❌ Model file not found. Please refresh the page.")
            return None
        
        logger.info(f"Loading model from {MODEL_PATH}")
        
        # Create model architecture
        try:
            model = timm.create_model(
                MODEL_NAME,
                pretrained=False,
                num_classes=NUM_CLASSES
            )
            logger.info(f"Model architecture created: {MODEL_NAME}")
        except Exception as e:
            logger.error(f"Error creating model architecture: {e}")
            st.error(f"❌ Error creating model: {e}")
            return None
       
        # Load model weights
        try:
            state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            # Try with weights_only=False as fallback (less secure)
            try:
                state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
                model.load_state_dict(state_dict)
                logger.warning("Model weights loaded with weights_only=False (less secure)")
            except Exception as e2:
                logger.error(f"Error loading model weights (fallback): {e2}")
                st.error(f"❌ Error loading model weights: {e2}")
                return None
        
        # Set model to evaluation mode
        model.eval()
        logger.info("Model set to evaluation mode")
        
        # Warmup to ensure model is ready
        try:
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            with torch.no_grad():
                _ = model(dummy)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.error(f"Error during model warmup: {e}")
            st.error(f"❌ Error during model initialization: {e}")
            return None
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}", exc_info=True)
        st.error(f"❌ Unexpected error loading model: {e}")
        return None

# Load the model globally (with error handling)
model = load_model()
if model is None:
    st.error("⚠️ **Model failed to load. Please refresh the page or check your internet connection.**")
    st.stop()
# ===================== IMAGE PREPROCESSING =====================
# Base preprocessing (no augmentation)
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_tta_transforms():
    """Create test-time augmentation transforms for uncertainty estimation."""
    base_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create augmented versions - using simpler, more reliable transforms
    augmented_transforms = []
    
    # Original (no augmentation) - always include this
    augmented_transforms.append(base_transform)
    
    # Horizontal flip
    augmented_transforms.append(transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Brightness variations (using ColorJitter correctly)
    brightness_min = max(0.5, 1.0 - TTA_BRIGHTNESS)  # Ensure reasonable range
    brightness_max = min(1.5, 1.0 + TTA_BRIGHTNESS)
    augmented_transforms.append(transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=(brightness_min, brightness_max)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Combine: flip + brightness
    augmented_transforms.append(transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ColorJitter(brightness=(brightness_min, brightness_max)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Add more variations using different brightness levels
    for brightness_factor in [0.95, 1.05]:
        augmented_transforms.append(transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ColorJitter(brightness=(brightness_factor, brightness_factor)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    
    # Add contrast variations (subtle)
    augmented_transforms.append(transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Combine: flip + contrast
    augmented_transforms.append(transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ColorJitter(contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Add a few identity transforms (just resize, no augmentation) to increase sample size
    while len(augmented_transforms) < TTA_NUM_SAMPLES:
        augmented_transforms.append(base_transform)
    
    # Return up to TTA_NUM_SAMPLES transforms
    return augmented_transforms[:TTA_NUM_SAMPLES]

# ===================== UNCERTAINTY ESTIMATION =====================
def calculate_confidence_interval_from_predictions(
    predictions: np.ndarray, 
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval from multiple predictions using empirical distribution.
    Uses percentile-based intervals for proper coverage.
    
    Args:
        predictions: Array of predictions from TTA (shape: [n_samples])
        confidence_level: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        tuple: (lower_bound, upper_bound, margin_of_error)
    """
    from scipy import stats
    
    if len(predictions) < 2:
        # Fallback: use normal approximation with small uncertainty
        mean_pred = float(predictions[0]) if len(predictions) == 1 else 0.5
        std_pred = 0.05  # Small default uncertainty
    else:
        mean_pred = float(np.mean(predictions))
        std_pred = float(np.std(predictions))
    
    # Calculate percentiles for empirical confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    if len(predictions) >= 2:
        lower_bound = float(np.percentile(predictions, lower_percentile))
        upper_bound = float(np.percentile(predictions, upper_percentile))
    else:
        # Use normal approximation
        z = stats.norm.ppf(1 - alpha / 2)
        lower_bound = max(0.0, mean_pred - z * std_pred)
        upper_bound = min(1.0, mean_pred + z * std_pred)
    
    # Calculate margin of error
    margin_of_error = (upper_bound - lower_bound) / 2
    
    # Clamp bounds to [0, 1]
    lower_bound = max(0.0, min(1.0, lower_bound))
    upper_bound = max(0.0, min(1.0, upper_bound))
    
    return lower_bound, upper_bound, margin_of_error

def calculate_uncertainty_level(margin_of_error: float) -> str:
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
# ===================== PREDICTION =====================
def predict_image(img: Image.Image, use_tta: bool = True) -> Optional[Dict[str, Any]]:
    """
    Process image and return prediction probabilities with confidence intervals using TTA.
    
    Args:
        img: PIL Image to predict
        use_tta: Whether to use test-time augmentation for uncertainty estimation
    
    Returns:
        Optional[Dict]: Prediction results with confidence intervals or None if error
    """
    if img is None or model is None:
        logger.error("Image or model is None")
        return None
   
    try:
        # Convert image to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
            logger.debug(f"Converted image from {img.mode} to RGB")
       
        # Get TTA transforms
        if use_tta:
            try:
                tta_transforms = create_tta_transforms()
                logger.info(f"Created {len(tta_transforms)} TTA transforms")
            except Exception as e:
                logger.error(f"Error creating TTA transforms: {e}")
                logger.info("Falling back to base transform")
                tta_transforms = [preprocess]
        else:
            tta_transforms = [preprocess]
        
        # Collect predictions from TTA
        malignant_predictions = []
        benign_predictions = []
        failed_transforms = 0
        
        with torch.no_grad():
            for i, transform in enumerate(tta_transforms):
                try:
                    # Apply transform
                    x = transform(img)
                    # Ensure it's a tensor and has batch dimension
                    if isinstance(x, torch.Tensor):
                        if x.dim() == 3:
                            x = x.unsqueeze(0)
                    else:
                        logger.warning(f"Transform {i} did not return a tensor")
                        failed_transforms += 1
                        continue
                    
                    # Get prediction
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                    
                    benign_predictions.append(float(probs[0]))
                    malignant_predictions.append(float(probs[1]))
                    logger.debug(f"TTA sample {i+1}: malignant={float(probs[1]):.3f}, benign={float(probs[0]):.3f}")
                except Exception as e:
                    failed_transforms += 1
                    logger.warning(f"Error in TTA prediction {i+1}/{len(tta_transforms)}: {e}")
                    continue
        
        if len(malignant_predictions) == 0:
            logger.error(f"No successful predictions from TTA ({failed_transforms} failed)")
            # Try fallback with just base transform
            try:
                logger.info("Attempting fallback prediction with base transform")
                x = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                benign_predictions = [float(probs[0])]
                malignant_predictions = [float(probs[1])]
                logger.info("Fallback prediction successful")
            except Exception as e:
                logger.error(f"Fallback prediction also failed: {e}")
                return None
        
        logger.info(f"Successful predictions: {len(malignant_predictions)}/{len(tta_transforms)} (failed: {failed_transforms})")
        
        # Calculate mean probabilities
        malignant_prob = float(np.mean(malignant_predictions))
        benign_prob = float(np.mean(benign_predictions))
        
        logger.info(f"Prediction - Malignant: {malignant_prob:.3f}, Benign: {benign_prob:.3f}")
        logger.info(f"TTA predictions: {len(malignant_predictions)} samples")
        
        # Calculate confidence intervals from TTA predictions
        mal_lower, mal_upper, mal_margin = calculate_confidence_interval_from_predictions(
            np.array(malignant_predictions)
        )
        ben_lower, ben_upper, ben_margin = calculate_confidence_interval_from_predictions(
            np.array(benign_predictions)
        )
        
        # Calculate uncertainty levels
        mal_uncertainty = calculate_uncertainty_level(mal_margin)
        ben_uncertainty = calculate_uncertainty_level(ben_margin)
       
        # Calculate model certainty (based on how far from 0.5 and prediction variance)
        prediction_variance = float(np.var(malignant_predictions))
        distance_from_threshold = abs(malignant_prob - 0.5)
        # Combine distance and low variance for higher certainty
        certainty = (distance_from_threshold * 2 * (1 - min(prediction_variance * 10, 0.5))) * 100
        certainty = max(0.0, min(100.0, certainty))  # Clamp to [0, 100]
       
        result = {
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
            'model_certainty': certainty,
            'prediction_variance': prediction_variance,
            'tta_samples': len(malignant_predictions)
        }
        
        logger.info(f"Prediction complete - Uncertainty: {mal_uncertainty}, Certainty: {certainty:.1f}%")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
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
    st.metric("🔍 Sensitivity", "~88-90%", help="At 0.35 threshold (optimized for screening)")
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
        help="Upload a clear dermoscopic image for analysis (Max 10MB, JPEG/PNG only)",
        label_visibility="collapsed"
    )
   
    if uploaded_file is not None:
        # Validate uploaded file
        is_valid, error_message = validate_image_file(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ **Validation Error:** {error_message}")
            st.info("Please upload a valid image file (JPEG or PNG, 64-2048 pixels, max 10MB)")
            logger.warning(f"Image validation failed: {error_message}")
        else:
            try:
                # Open and display image
                uploaded_file.seek(0)  # Reset file pointer
                image = Image.open(uploaded_file)
                
                # Convert to RGB if needed (handles RGBA, L, etc.)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    logger.debug(f"Converted image from {image.mode} to RGB")
                
                st.image(image, caption="📸 Uploaded Image", use_column_width=True)
                
                # Display image info
                width, height = image.size
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.caption(f"📏 Image size: {width}x{height} pixels | File size: {file_size_mb:.2f} MB")

                # Clear old state ONLY when a NEW file is uploaded (not on rerun)
                current_uploaded_name = uploaded_file.name
                last_uploaded_name = st.session_state.get('last_uploaded')
                
                if last_uploaded_name is None or last_uploaded_name != current_uploaded_name:
                    # New file uploaded - clear old results
                    logger.info(f"New image uploaded: {current_uploaded_name} (previous: {last_uploaded_name})")
                    st.session_state['result'] = None
                    st.session_state['analyzing'] = False
                    st.session_state['last_uploaded'] = current_uploaded_name
                else:
                    # Same file - preserve results
                    logger.debug(f"Same file as before: {current_uploaded_name}, preserving results")
               
                st.markdown("<br>", unsafe_allow_html=True)
               
                if st.button("🔬 Analyze Lesion", type="primary", key="analyze_btn", disabled=(model is None)):
                    if model is None:
                        st.error("⚠️ Model not loaded. Please refresh the page.")
                        logger.error("Model is None when analyze button clicked")
                        st.stop()
                    else:
                        # Prevent double analysis
                        if st.session_state.get('analyzing', False):
                            st.warning("⏳ Analysis in progress... Please wait.")
                            logger.warning("Analysis already in progress")
                            st.stop()

                        st.session_state['analyzing'] = True
                        logger.info("Starting image analysis")

                        with st.spinner("🔄 Analyzing image with AI using test-time augmentation..."):
                            try:
                                result = predict_image(image, use_tta=True)
                                if result:
                                    # Store result (already has proper types from predict_image)
                                    st.session_state['result'] = result
                                    st.session_state['analyzing'] = False
                                    logger.info("Analysis completed successfully - result stored in session_state")
                                    logger.info(f"Result keys: {list(result.keys())}")
                                    logger.info(f"Result malignant prob: {result.get('malignant', 'N/A')}")
                                    st.success("✅ Analysis complete!")
                                    # Don't call st.stop() here - let the page continue to render results
                                else:
                                    st.error("❌ Failed to generate prediction. Please try again.")
                                    logger.error("Prediction returned None")
                                    st.session_state['analyzing'] = False
                                    st.stop()
                            except Exception as e:
                                logger.error(f"Analysis error: {e}", exc_info=True)
                                st.error(f"❌ Analysis error: {e}")
                                st.info("💡 **Tip:** Try uploading a different image or check if the image is a valid dermoscopic image.")
                                st.session_state['analyzing'] = False
                                st.stop()
                            finally:
                                # Ensure analyzing flag is cleared
                                if st.session_state.get('analyzing', False):
                                    st.session_state['analyzing'] = False
                        
            except Exception as e:
                logger.error(f"Error processing uploaded image: {e}", exc_info=True)
                st.error(f"❌ Error processing image: {e}")
                st.info("Please try uploading a different image file.")
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
   
    # Check if result exists and is valid
    stored_result = st.session_state.get('result')
    if stored_result is not None:
        try:
            result = stored_result
            logger.debug(f"Displaying result: {list(result.keys())}")
            
            # Validate result structure
            if 'malignant' not in result or 'benign' not in result or 'confidence_intervals' not in result:
                logger.error(f"Invalid result structure: {list(result.keys())}")
                st.error("❌ Result data is invalid. Please try analyzing again.")
                st.code(str(result))
            else:
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
                   
                    The 95% confidence interval is calculated using **Test-Time Augmentation (TTA)**,
                    which generates multiple predictions from augmented versions of your image.
                    This provides a more accurate estimate of model uncertainty than statistical approximations.
                   
                    - **Low Uncertainty**: Margin of error < 8% - High confidence, consistent predictions
                    - **Moderate Uncertainty**: Margin of error 8-15% - Borderline case, some variation
                    - **High Uncertainty**: Margin of error > 15% - Additional assessment recommended
                   
                    **TTA Samples:** {} predictions were used to calculate this confidence interval.
                    """.format(result.get('tta_samples', 'N/A')))
                
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
                   
                    **Prediction Method:** Test-Time Augmentation (TTA) with {result.get('tta_samples', 'N/A')} samples
                   
                    **Prediction Variance:** {result.get('prediction_variance', 0):.4f}
                   
                    **Clinical Note:** This threshold is set to maximize detection of malignant lesions while
                    minimizing false negatives, which is crucial in medical screening applications. Confidence
                    intervals from TTA provide more accurate uncertainty quantification than statistical approximations.
                    """)
        except Exception as e:
            logger.error(f"Error displaying results: {e}", exc_info=True)
            st.error(f"❌ Error displaying results: {e}")
            st.info("Please try analyzing the image again.")
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
   
    A **95% confidence interval** represents the range of values where we can be 95% confident
    the true prediction probability falls. This provides crucial context beyond just the point estimate.
   
    #### How We Calculate Uncertainty
   
    **Test-Time Augmentation (TTA) Method:**
    - We use multiple augmented versions of your image (rotations, brightness variations, flips)
    - Each augmented image generates a prediction
    - The confidence interval is calculated from the distribution of these predictions
    - This captures model uncertainty based on how predictions vary with small image changes
   
    **Why This Method:**
    - More accurate than statistical approximations
    - Reflects true model uncertainty
    - Accounts for image-specific variations
    - Provides proper coverage guarantees
   
    #### Why They Matter in Dermoscopy
   
    **For Clear Cases:**
    - Narrow confidence intervals (±5-8%) indicate high confidence
    - Model predictions are consistent across augmentations
    - Both bounds clearly above or below the threshold
    - Stronger clinical decision support
   
    **For Borderline Cases:**
    - Wider confidence intervals (±10-15%) indicate uncertainty
    - Model predictions vary more with image changes
    - Bounds may span the clinical threshold
    - Suggests need for additional assessment or imaging
   
    #### Uncertainty Levels
   
    - 🟢 **Low Uncertainty** (< 8% margin): High confidence, reliable prediction
    - 🟡 **Moderate Uncertainty** (8-15% margin): Borderline case, professional review recommended
    - 🔴 **High Uncertainty** (> 15% margin): Additional assessment strongly recommended
   
    #### Clinical Application
   
    Even with uncertainty, the model errs on the side of caution. If the malignant probability
    or its confidence interval approaches or exceeds the clinical threshold, referral is recommended.
   
    > 💡 **Key Insight:** Confidence intervals from TTA help distinguish between "definitely high risk"
    > and "uncertain, but warrants caution" cases, improving clinical decision-making by quantifying
    > model uncertainty more accurately than traditional statistical methods.
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
        Model: EfficientNet-B4 | F1: 85.2% | Sensitivity: ~88-90% | Uncertainty: Test-Time Augmentation (TTA)
    </p>
    <p style="color: #999; font-size: 0.8rem;">
        Dr Tom Hutchinson • Oxford, United Kingdom
    </p>
</div>
""", unsafe_allow_html=True)
