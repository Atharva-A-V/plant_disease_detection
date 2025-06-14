#!/usr/bin/env python3
"""
Test the actual model predictions and see what class names are being returned
"""

import torch
import torch.nn as nn
import torchvision.models as models
import json
from pathlib import Path
import numpy as np

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

print(f"Loaded {len(class_names)} class names")

# Test what the model actually returns
device = torch.device("cpu")

# Try to load the model
model_paths = [
    "plant_disease_model_complete.pth",
    "best_plant_disease_model.pth"
]

for model_path in model_paths:
    model_file = Path(model_path)
    if model_file.exists():
        print(f"\nTesting model: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_file, map_location=device)
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
        
        # Check classifier layer
        classifier_keys = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]
        if classifier_keys:
            final_layer_key = max(classifier_keys, key=lambda x: int(x.split('.')[1]) if x.split('.')[1].isdigit() else 0)
            actual_num_classes = state_dict[final_layer_key].shape[0]
            print(f"Model outputs {actual_num_classes} classes")
            
            # Create model
            model = models.efficientnet_b3(weights=None)
            num_features = model.classifier[1].in_features
            
            if actual_num_classes == 2:
                model.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_features, actual_num_classes)
                )
                print("Using binary classifier")
            else:
                model.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, actual_num_classes)
                )
                print("Using multi-class classifier")
            
            # Load weights
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            # Test with dummy input
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                output = model(dummy_input)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                
                print(f"Output shape: {output.shape}")
                print(f"Probabilities shape: {probabilities.shape}")
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probabilities[0], min(5, len(class_names)))
                
                print(f"\nTop 5 predictions:")
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    idx = idx.item()
                    prob = prob.item()
                    if idx < len(class_names):
                        class_name = class_names[idx]
                        
                        # Test healthy detection on this class
                        parts = class_name.split('___')
                        disease = parts[1].replace('_', ' ').title() if len(parts) > 1 else "Unknown"
                        is_healthy = 'healthy' in class_name.lower() or 'healthy' in disease.lower()
                        status = "HEALTHY" if is_healthy else "DISEASED"
                        
                        print(f"  {i+1}. Index {idx}: {class_name} -> {status} ({prob:.3f})")
                    else:
                        print(f"  {i+1}. Index {idx}: OUT OF RANGE ({prob:.3f})")
        
        break