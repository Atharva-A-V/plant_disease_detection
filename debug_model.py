#!/usr/bin/env python3
"""
Debug script to test model predictions and identify why healthy plants are being classified as diseased.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np

def load_model_and_test():
    """Load the model and test its predictions"""
    
    # Load class names
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    
    print(f"Loaded {len(class_names)} classes")
    
    # Find healthy classes
    healthy_indices = []
    for i, name in enumerate(class_names):
        if 'healthy' in name.lower():
            healthy_indices.append(i)
            print(f"Healthy class {i}: {name}")
    
    print(f"\nFound {len(healthy_indices)} healthy classes")
    
    # Load model
    device = torch.device("cpu")
    model = models.efficientnet_b3(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, len(class_names))
    )
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load("best_plant_disease_model.pth", map_location=device))
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    model.eval()
    
    # Test with a dummy image (random noise - should give random predictions)
    print("\n🧪 Testing with random noise image...")
    dummy_image = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_image)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[predicted_idx].item()
    
    print(f"Prediction: Class {predicted_idx} ({class_names[predicted_idx]})")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Is healthy class: {'healthy' in class_names[predicted_idx].lower()}")
    
    # Check if model is biased towards certain classes
    print("\n📊 Top 10 predictions for random input:")
    top_indices = torch.argsort(probabilities, descending=True)[:10]
    
    healthy_in_top10 = 0
    for i, idx in enumerate(top_indices):
        is_healthy = 'healthy' in class_names[idx].lower()
        if is_healthy:
            healthy_in_top10 += 1
        print(f"{i+1:2d}. Class {idx:2d}: {class_names[idx]:<40} {probabilities[idx]*100:5.2f}% {'✅' if is_healthy else '❌'}")
    
    print(f"\nHealthy classes in top 10: {healthy_in_top10}/10")
    
    # Check distribution of healthy vs diseased probabilities
    healthy_prob_sum = sum(probabilities[i] for i in healthy_indices)
    diseased_prob_sum = 1.0 - healthy_prob_sum
    
    print(f"\n🎯 Probability Distribution:")
    print(f"   Total healthy probability: {healthy_prob_sum*100:.2f}%")
    print(f"   Total diseased probability: {diseased_prob_sum*100:.2f}%")
    
    # Test the get_plant_info function logic
    print(f"\n🔍 Testing get_plant_info logic for predicted class:")
    predicted_class = class_names[predicted_idx]
    parts = predicted_class.split('___')
    plant = parts[0].replace('_', ' ').title()
    disease = parts[1].replace('_', ' ').title() if len(parts) > 1 else "Unknown"
    is_healthy_detected = 'healthy' in disease.lower()
    
    print(f"   Raw class: {predicted_class}")
    print(f"   Plant: {plant}")
    print(f"   Disease: {disease}")
    print(f"   Is healthy detected: {is_healthy_detected}")
    
    return model, class_names, healthy_indices

if __name__ == "__main__":
    print("🌱 Plant Disease Detection - Debug Script")
    print("=" * 50)
    load_model_and_test()