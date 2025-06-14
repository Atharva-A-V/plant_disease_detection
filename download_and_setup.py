#!/usr/bin/env python3
"""
Download PlantVillage dataset using kagglehub and prepare for training
"""

import kagglehub
import os
import shutil
from pathlib import Path

def download_dataset():
    """Download the PlantVillage dataset using kagglehub"""
    print("📥 Downloading PlantVillage dataset using kagglehub...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("emmarex/plantdisease")
        print("Path to dataset files:", path)
        
        # Check if the dataset was downloaded successfully
        dataset_path = Path(path)
        if dataset_path.exists():
            print(f"✅ Dataset downloaded successfully to: {path}")
            
            # List the contents to understand the structure
            print("\n📁 Dataset structure:")
            for item in dataset_path.rglob("*"):
                if item.is_file() and str(item).endswith(('.jpg', '.jpeg', '.png')):
                    print(f"   📄 {item.relative_to(dataset_path)}")
                elif item.is_dir():
                    print(f"   📂 {item.relative_to(dataset_path)}/")
                    
            return str(dataset_path)
        else:
            print("❌ Dataset download failed - path doesn't exist")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def setup_training_structure(dataset_path):
    """Set up the training directory structure"""
    print(f"\n🔧 Setting up training structure from: {dataset_path}")
    
    source_path = Path(dataset_path)
    
    # Find the actual image directory
    image_dirs = []
    for item in source_path.rglob("*"):
        if item.is_dir() and any(f.suffix.lower() in ['.jpg', '.jpeg', '.png'] for f in item.iterdir() if f.is_file()):
            image_dirs.append(item)
    
    if not image_dirs:
        print("❌ No image directories found in the dataset")
        return None
    
    # Use the first directory with images
    main_image_dir = image_dirs[0]
    print(f"📂 Found image directory: {main_image_dir}")
    
    # Create local training directory
    training_dir = Path("PlantVillage-Dataset/raw/color")
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy or link the dataset
    if not (training_dir / "copied").exists():
        print("📋 Copying dataset to local training directory...")
        
        # Copy each class directory
        class_count = 0
        for class_dir in main_image_dir.iterdir():
            if class_dir.is_dir():
                dest_dir = training_dir / class_dir.name
                if not dest_dir.exists():
                    shutil.copytree(class_dir, dest_dir)
                    class_count += 1
                    print(f"   ✅ Copied {class_dir.name}")
        
        # Mark as copied
        (training_dir / "copied").touch()
        print(f"✅ Copied {class_count} class directories")
    else:
        print("✅ Dataset already copied to training directory")
    
    return str(training_dir)

if __name__ == "__main__":
    print("🚀 DOWNLOADING PLANTVILLAGE DATASET FOR 100-EPOCH TRAINING")
    print("="*70)
    
    # Download the dataset
    dataset_path = download_dataset()
    
    if dataset_path:
        # Set up training structure
        training_path = setup_training_structure(dataset_path)
        
        if training_path:
            print(f"\n✅ Dataset ready for training!")
            print(f"📁 Training data path: {training_path}")
            print("\n🚀 Starting 100-epoch training...")
            print("="*50)
        else:
            print("❌ Failed to set up training structure")
    else:
        print("❌ Failed to download dataset")