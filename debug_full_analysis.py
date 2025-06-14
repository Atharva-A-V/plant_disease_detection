#!/usr/bin/env python3
"""
Simple test to check model predictions and class mapping
"""

import json
import random

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

print("=== CLASS NAME ANALYSIS ===")
print(f"Total classes: {len(class_names)}")

# Check healthy vs diseased classes
healthy_classes = []
diseased_classes = []

for i, class_name in enumerate(class_names):
    is_healthy = 'healthy' in class_name.lower()
    
    if is_healthy:
        healthy_classes.append((i, class_name))
    else:
        diseased_classes.append((i, class_name))

print(f"\nHealthy classes ({len(healthy_classes)}):")
for idx, name in healthy_classes:
    print(f"  Index {idx}: {name}")

print(f"\nDiseased classes ({len(diseased_classes)}):")
for idx, name in diseased_classes[:5]:  # Show first 5
    print(f"  Index {idx}: {name}")
print(f"  ... and {len(diseased_classes)-5} more")

# Test the get_plant_info logic on actual class names
print(f"\n=== TESTING GET_PLANT_INFO LOGIC ===")

def test_get_plant_info_logic(class_name):
    parts = class_name.split('___')
    plant = parts[0].replace('_', ' ').title()
    disease = parts[1].replace('_', ' ').title() if len(parts) > 1 else "Unknown"
    is_healthy = 'healthy' in class_name.lower() or 'healthy' in disease.lower()
    
    return plant, disease, is_healthy

# Test a few examples
test_cases = [
    class_names[3],   # Should be Apple___healthy
    class_names[0],   # Should be Apple___Apple_scab  
    class_names[15],  # Should be Grape___healthy
    class_names[12],  # Should be some disease
]

for class_name in test_cases:
    plant, disease, is_healthy = test_get_plant_info_logic(class_name)
    status = "✅ HEALTHY" if is_healthy else "❌ DISEASED"
    print(f"{class_name:<45} -> {status}")

# Check if there might be a binary model issue
print(f"\n=== CHECKING FOR BINARY MODEL ===")
try:
    with open('binary_class_names.json', 'r') as f:
        binary_names = json.load(f)
    print(f"Binary model detected with classes: {binary_names}")
    print("This might be the issue - app could be using binary model!")
except:
    print("No binary model found")

print(f"\n=== POTENTIAL ISSUES ===")
print("1. Check if app is loading the wrong model (binary vs multi-class)")
print("2. Check if model predictions are mapping to wrong indices")
print("3. Check if there's a caching issue in Streamlit")
print("4. Verify the model file matches the class names")