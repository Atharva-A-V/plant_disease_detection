#!/usr/bin/env python3
"""
Direct test of model loading and prediction to identify the exact issue
"""

import torch
import torch.nn as nn
import torchvision.models as models
import json
from pathlib import Path
import numpy as np

print("=== TESTING MODEL LOADING AND PREDICTION ===")

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

print(f"Class names loaded: {len(class_names)} classes")

# Test model loading
device = torch.device("cpu")
model_path = "best_plant_disease_model.pth"

print(f"\nTesting model: {model_path}")

# Load and inspect the model
checkpoint = torch.load(model_path, map_location=device)

if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Check classifier layer
classifier_keys = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]
print(f"Classifier keys found: {classifier_keys}")

if classifier_keys:
    final_layer_key = max(classifier_keys, key=lambda x: int(x.split('.')[1]) if x.split('.')[1].isdigit() else 0)
    actual_num_classes = state_dict[final_layer_key].shape[0]
    print(f"Model output classes: {actual_num_classes}")
    print(f"Final layer: {final_layer_key} -> {state_dict[final_layer_key].shape}")

# Create model
model = models.efficientnet_b3(weights=None)
num_features = model.classifier[1].in_features

# Build classifier
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, actual_num_classes)
)

# Load weights
try:
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Test prediction
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    
    print(f"\nPrediction test:")
    print(f"Output shape: {output.shape}")
    print(f"Raw output range: {output.min():.3f} to {output.max():.3f}")
    print(f"Probabilities range: {probabilities.min():.3f} to {probabilities.max():.3f}")
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities[0], 5)
    
    print(f"\nTop 5 predictions:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        idx = idx.item()
        prob = prob.item()
        if idx < len(class_names):
            class_name = class_names[idx]
            is_healthy = 'healthy' in class_name.lower()
            status = "HEALTHY" if is_healthy else "DISEASED"
            print(f"  {i+1}. {class_name} -> {status} ({prob*100:.1f}%)")

print(f"\n=== ISSUE DIAGNOSIS ===")
print("If the model shows very low confidence (like 2.8%), it suggests:")
print("1. Model architecture mismatch with saved weights")
print("2. Model not trained properly") 
print("3. Input preprocessing issues")
print("4. Model outputs are not meaningful")