#!/usr/bin/env python3
"""
Simple dataset setup for training
"""

import os
import json
from pathlib import Path

def create_training_dataset():
    """Create a simple dataset structure for training"""
    print("🧪 Creating training dataset...")
    
    # Load existing class names
    try:
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
    except:
        class_names = [
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
    
    # Create sample dataset directory structure
    sample_dir = Path("training_dataset")
    sample_dir.mkdir(exist_ok=True)
    
    print(f"Creating {len(class_names)} class directories...")
    for class_name in class_names:
        class_dir = sample_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create dummy image files (just empty files for structure)
        for i in range(50):  # 50 samples per class
            dummy_file = class_dir / f"{class_name}_{i:03d}.jpg"
            dummy_file.touch()
    
    print(f"✅ Training dataset structure created with {len(class_names)} classes")
    return str(sample_dir)

if __name__ == "__main__":
    dataset_path = create_training_dataset()
    print(f"📁 Dataset path: {dataset_path}")
    print("✅ Ready to start training!")