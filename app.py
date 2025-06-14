import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
import logging
from pathlib import Path
import gc
import warnings

# Suppress warnings for cleaner deployment
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🌱 Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .healthy-box {
        border-left-color: #4CAF50;
        background-color: #e8f5e8;
    }
    .diseased-box {
        border-left-color: #f44336;
        background-color: #ffeaea;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stSpinner > div {
        border-top-color: #2E7D32;
    }
</style>
""", unsafe_allow_html=True)

# Default class names (fallback)
DEFAULT_CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def validate_image(image_file):
    """Validate uploaded image file"""
    try:
        # Check file size (limit to 10MB)
        if len(image_file.getvalue()) > 10 * 1024 * 1024:
            return False, "File size too large. Please upload an image smaller than 10MB."
        
        # Try to open the image
        image = Image.open(image_file)
        
        # Check image dimensions
        if image.size[0] < 50 or image.size[1] < 50:
            return False, "Image too small. Please upload an image larger than 50x50 pixels."
        
        if image.size[0] > 5000 or image.size[1] > 5000:
            return False, "Image too large. Please upload an image smaller than 5000x5000 pixels."
        
        return True, "Valid image"
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

@st.cache_resource
def load_class_names():
    """Load class names from file or use defaults"""
    try:
        class_names_path = Path('class_names.json')
        if class_names_path.exists():
            with open(class_names_path, 'r') as f:
                class_names = json.load(f)
                logger.info(f"Loaded {len(class_names)} class names from file")
                return class_names
    except Exception as e:
        logger.warning(f"Could not load class names from file: {e}")
    
    logger.info(f"Using default class names: {len(DEFAULT_CLASS_NAMES)} classes")
    return DEFAULT_CLASS_NAMES

@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    try:
        device = torch.device("cpu")  # Force CPU for deployment
        class_names = load_class_names()
        
        # Load the model architecture
        model = models.efficientnet_b3(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, len(class_names))
        )
        
        # Try to load the trained weights
        model_paths = [
            "best_plant_disease_model.pth",
            "plant_disease_model_complete.pth",
            "potato_disease_model.pth"
        ]
        
        model_loaded = False
        for model_path in model_paths:
            model_file = Path(model_path)
            if model_file.exists():
                try:
                    if model_path == "plant_disease_model_complete.pth":
                        checkpoint = torch.load(model_file, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(torch.load(model_file, map_location=device))
                    
                    model_loaded = True
                    logger.info(f"Model loaded successfully from {model_path}")
                    st.success(f"✅ Model loaded from {model_path}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load model from {model_path}: {e}")
                    continue
        
        if not model_loaded:
            error_msg = "❌ No trained model found. Please ensure a trained model file is available."
            logger.error(error_msg)
            st.error(error_msg)
            return None, None
            
        model.eval()
        
        # Enable memory optimization
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
        
        return model, device
        
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None, None

def preprocess_image(image):
    """Preprocess image for model prediction with error handling"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        processed_image = transform(image).unsqueeze(0)
        return processed_image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def predict_disease(model, image, device):
    """Make prediction on the image with error handling"""
    if model is None:
        return None, None, None
    
    try:
        processed_image = preprocess_image(image)
        processed_image = processed_image.to(device)
        
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
        
        # Clear GPU memory if using GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Force garbage collection for memory management
        gc.collect()
        
        return predicted.item(), confidence.item(), probabilities.cpu().numpy()
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def get_plant_info(class_name):
    """Get plant and disease information with enhanced error handling"""
    try:
        parts = class_name.split('___')
        plant = parts[0].replace('_', ' ').title()
        disease = parts[1].replace('_', ' ').title() if len(parts) > 1 else "Unknown"
        
        # Enhanced disease information database
        disease_info = {
            "healthy": {
                "description": "The plant appears to be healthy with no visible signs of disease.",
                "treatment": "Continue regular care and monitoring. Maintain proper watering, fertilization, and pest control.",
                "severity": "None",
                "prevention": "Keep plants well-spaced, ensure good air circulation, and monitor regularly.",
                "urgency": "Low"
            },
            "apple scab": {
                "description": "A fungal disease causing dark, scabby lesions on leaves and fruit.",
                "treatment": "Apply fungicide (captan or myclobutanil), improve air circulation, remove infected leaves.",
                "severity": "Moderate",
                "prevention": "Plant resistant varieties, avoid overhead watering, rake and dispose of fallen leaves.",
                "urgency": "Medium"
            },
            "late blight": {
                "description": "A serious disease causing brown lesions and potential crop loss.",
                "treatment": "Apply copper-based fungicide immediately, ensure proper drainage, remove infected plants.",
                "severity": "High",
                "prevention": "Avoid overhead watering, provide good air circulation, use certified disease-free seeds.",
                "urgency": "High"
            },
            "early blight": {
                "description": "Fungal disease causing dark spots with concentric rings on leaves.",
                "treatment": "Apply fungicide (chlorothalonil or copper), improve air circulation, water at soil level.",
                "severity": "Moderate",
                "prevention": "Rotate crops, mulch around plants, avoid overhead watering.",
                "urgency": "Medium"
            },
            "black rot": {
                "description": "Fungal disease causing circular brown spots on leaves and fruit rot.",
                "treatment": "Apply fungicide, prune affected areas, improve air circulation.",
                "severity": "High",
                "prevention": "Remove mummified fruit, prune for air circulation, apply preventive fungicides.",
                "urgency": "High"
            },
            "powdery mildew": {
                "description": "White powdery coating on leaves, stems, and sometimes fruit.",
                "treatment": "Apply sulfur or potassium bicarbonate spray, improve air circulation.",
                "severity": "Moderate",
                "prevention": "Avoid overhead watering, space plants properly, choose resistant varieties.",
                "urgency": "Medium"
            }
        }
        
        disease_lower = disease.lower()
        info = None
        
        # Check for partial matches
        for key in disease_info.keys():
            if key in disease_lower or disease_lower in key:
                info = disease_info[key]
                break
        
        if info is None:
            info = {
                "description": f"Disease detected: {disease}",
                "treatment": "Consult with a plant pathologist or agricultural extension service for specific treatment recommendations.",
                "severity": "Unknown",
                "prevention": "Follow general plant care practices and monitor regularly.",
                "urgency": "Unknown"
            }
        
        return plant, disease, info
        
    except Exception as e:
        logger.error(f"Error getting plant info for {class_name}: {e}")
        return "Unknown", "Unknown", {
            "description": "Unable to determine plant information",
            "treatment": "Please consult with a plant expert",
            "severity": "Unknown",
            "prevention": "Follow general plant care practices",
            "urgency": "Unknown"
        }

def main():
    try:
        # Header
        st.markdown('<h1 class="main-header">🌱 Plant Disease Detection System</h1>', unsafe_allow_html=True)
        
        # Load model and class names
        model, device = load_model()
        class_names = load_class_names()
        
        # Sidebar
        st.sidebar.markdown("## 📋 About")
        st.sidebar.info(
            "This AI-powered system can detect diseases in plant leaves using deep learning. "
            "Upload an image of a plant leaf to get instant disease detection and treatment recommendations."
        )
        
        st.sidebar.markdown("## 🔧 Model Info")
        st.sidebar.write("- **Architecture**: EfficientNet-B3")
        st.sidebar.write(f"- **Classes**: {len(class_names)} plant diseases")
        st.sidebar.write("- **Input Size**: 224x224 pixels")
        st.sidebar.write("- **Framework**: PyTorch")
        
        st.sidebar.markdown("## 📚 Supported Plants")
        plants = set()
        for class_name in class_names:
            plant = class_name.split('___')[0].replace('_', ' ').title()
            plants.add(plant)
        
        for plant in sorted(plants):
            st.sidebar.write(f"• {plant}")
        
        # Main content
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Upload Plant Image")
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload a clear image of a plant leaf for disease detection"
            )
            
            if uploaded_file is not None:
                # Validate image
                is_valid, validation_message = validate_image(uploaded_file)
                
                if not is_valid:
                    st.error(f"❌ {validation_message}")
                    return
                
                # Display uploaded image
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Image info
                    with st.expander("📊 Image Details"):
                        st.write(f"**Size**: {image.size[0]} x {image.size[1]} pixels")
                        st.write(f"**Mode**: {image.mode}")
                        st.write(f"**Format**: {getattr(image, 'format', 'Unknown')}")
                        st.write(f"**File size**: {len(uploaded_file.getvalue())/1024:.1f} KB")
                        
                except Exception as e:
                    st.error(f"❌ Error loading image: {str(e)}")
                    return
        
        with col2:
            if uploaded_file is not None and model is not None and 'image' in locals():
                st.markdown("### 🔍 Analysis Results")
                
                with st.spinner("🔄 Analyzing image..."):
                    # Make prediction
                    predicted_idx, confidence, probabilities = predict_disease(model, image, device)
                    
                    if predicted_idx is not None:
                        predicted_class = class_names[predicted_idx]
                        plant, disease, info = get_plant_info(predicted_class)
                        
                        # Prediction result
                        is_healthy = "healthy" in disease.lower()
                        box_class = "healthy-box" if is_healthy else "diseased-box"
                        
                        st.markdown(f"""
                        <div class="prediction-box {box_class}">
                            <h3>🌿 Plant: {plant}</h3>
                            <h4>{'✅' if is_healthy else '⚠️'} Status: {disease}</h4>
                            <p><strong>Confidence:</strong> {confidence*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence meter
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = confidence*100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Confidence Level"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#4CAF50" if is_healthy else "#f44336"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "#FFC107"},
                                    {'range': [80, 100], 'color': "#4CAF50"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=300)
                        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Detailed results section
        if uploaded_file is not None and model is not None and 'predicted_idx' in locals() and predicted_idx is not None:
            st.markdown("---")
            
            col3, col4 = st.columns([1, 1])
            
            with col3:
                st.markdown("### 📊 Top Predictions")
                
                # Get top 5 predictions
                top_indices = np.argsort(probabilities)[-5:][::-1]
                top_classes = [class_names[i] for i in top_indices]
                top_probs = [probabilities[i] * 100 for i in top_indices]
                
                # Create DataFrame for better display
                df_predictions = pd.DataFrame({
                    'Disease': [get_plant_info(cls)[1] for cls in top_classes],
                    'Plant': [get_plant_info(cls)[0] for cls in top_classes],
                    'Confidence (%)': [round(prob, 1) for prob in top_probs]
                })
                
                # Display as bar chart
                fig_bar = px.bar(
                    df_predictions, 
                    x='Confidence (%)', 
                    y='Disease',
                    orientation='h',
                    title="Top 5 Predictions",
                    color='Confidence (%)',
                    color_continuous_scale='RdYlGn',
                    text='Confidence (%)'
                )
                fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_bar.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Display as table
                st.dataframe(df_predictions, use_container_width=True)
            
            with col4:
                st.markdown("### 💡 Recommendations")
                
                predicted_class = class_names[predicted_idx]
                plant, disease, info = get_plant_info(predicted_class)
                
                # Description
                st.markdown("#### 📝 Description")
                st.write(info['description'])
                
                # Severity and Urgency
                severity_color = {
                    "None": "🟢",
                    "Low": "🟡", 
                    "Moderate": "🟠",
                    "High": "🔴",
                    "Unknown": "⚪"
                }
                severity_icon = severity_color.get(info['severity'], "⚪")
                st.markdown(f"#### {severity_icon} Severity Level: {info['severity']}")
                
                if 'urgency' in info and info['urgency'] != 'Unknown':
                    urgency_icon = severity_color.get(info['urgency'], "⚪")
                    st.markdown(f"#### {urgency_icon} Action Urgency: {info['urgency']}")
                
                # Treatment
                st.markdown("#### 🔬 Treatment")
                st.write(info['treatment'])
                
                # Prevention
                if 'prevention' in info:
                    st.markdown("#### 🛡️ Prevention")
                    st.write(info['prevention'])
                
                # Alert
                if not is_healthy:
                    st.error("⚠️ Disease detected! Please take appropriate action based on the recommendations above.")
                    st.markdown("**📞 Consider consulting with:**")
                    st.write("• Agricultural extension service")
                    st.write("• Plant pathologist")
                    st.write("• Local nursery specialist")
                else:
                    st.success("✅ Plant appears healthy! Continue regular care and monitoring.")
        
        # Additional features
        if model is not None:
            st.markdown("---")
            
            # Model performance section
            with st.expander("📈 Model Performance Information"):
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    st.metric("Model Architecture", "EfficientNet-B3")
                
                with col6:
                    st.metric("Total Classes", len(class_names))
                    
                with col7:
                    st.metric("Input Resolution", "224×224")
            
            # Tips section
            with st.expander("💡 Tips for Better Results"):
                st.markdown("""
                **📸 Image Quality Tips:**
                - Use good lighting (natural light preferred)
                - Capture the entire leaf if possible
                - Avoid blurry or out-of-focus images
                - Include symptoms clearly in the frame
                - Use a plain background when possible
                
                **🎯 Best Practices:**
                - Take multiple photos from different angles
                - Capture both affected and healthy parts
                - Ensure the disease symptoms are clearly visible
                - Use high-resolution images when possible
                - File size should be under 10MB
                """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "🌱 Plant Disease Detection System | Powered by EfficientNet & Streamlit<br>"
            "Made with ❤️ for sustainable agriculture"
            "</div>", 
            unsafe_allow_html=True
        )
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()