#!/usr/bin/env python3
"""
Download PlantVillage dataset and start training
"""

import os
import requests
import zipfile
from pathlib import Path
import kaggle

def download_plantvillage_dataset():
    """Download the PlantVillage dataset"""
    print("📥 Downloading PlantVillage dataset...")
    
    dataset_dir = Path("PlantVillage-Dataset")
    
    if dataset_dir.exists():
        print("✅ Dataset already exists!")
        return str(dataset_dir / "raw" / "color")
    
    try:
        # Try using Kaggle API first
        print("🔄 Attempting to download via Kaggle API...")
        os.system("kaggle datasets download -d abdallahalidev/plantvillage-dataset")
        
        # Extract the zip file
        with zipfile.ZipFile("plantvillage-dataset.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Clean up
        os.remove("plantvillage-dataset.zip")
        
        print("✅ Dataset downloaded successfully!")
        return str(dataset_dir / "raw" / "color")
        
    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        print("📝 Please manually download the dataset from:")
        print("   https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("   and extract it to the current directory")
        return None

def create_sample_dataset():
    """Create a sample dataset structure for testing"""
    print("🧪 Creating sample dataset for testing...")
    
    # Load existing class names
    try:
        import json
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
    sample_dir = Path("sample_dataset")
    sample_dir.mkdir(exist_ok=True)
    
    # Create dummy images for each class (for structure only)
    from PIL import Image
    import numpy as np
    
    print(f"Creating {len(class_names)} class directories...")
    for class_name in class_names:
        class_dir = sample_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create 10 dummy images per class
        for i in range(10):
            # Create a random colored image
            if 'healthy' in class_name.lower():
                # Green-ish images for healthy plants
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img_array[:, :, 1] = np.clip(img_array[:, :, 1] + 50, 0, 255)  # More green
            else:
                # Brown-ish images for diseased plants  
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 30, 0, 255)  # More red/brown
                img_array[:, :, 1] = np.clip(img_array[:, :, 1] - 30, 0, 255)  # Less green
            
            img = Image.fromarray(img_array)
            img.save(class_dir / f"{class_name}_{i:03d}.jpg")
    
    print(f"✅ Sample dataset created with {len(class_names)} classes")
    return str(sample_dir)

if __name__ == "__main__":
    print("🚀 DATASET SETUP FOR 100-EPOCH TRAINING")
    print("="*50)
    
    # Try to download real dataset first
    dataset_path = download_plantvillage_dataset()
    
    if dataset_path is None:
        print("\n⚠️ Real dataset not available, creating sample dataset...")
        dataset_path = create_sample_dataset()
    
    print(f"\n📁 Dataset path: {dataset_path}")
    print("✅ Ready to start training!")
    print("\nTo start training, run:")
    print(f"python train.py --data_path {dataset_path}")