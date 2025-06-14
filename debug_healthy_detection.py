#!/usr/bin/env python3
"""
Debug script to test healthy detection logic
"""

import json

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

def test_get_plant_info(class_name):
    """Test the get_plant_info function logic"""
    print(f"\n=== Testing: {class_name} ===")
    
    parts = class_name.split('___')
    plant = parts[0].replace('_', ' ').title()
    disease = parts[1].replace('_', ' ').title() if len(parts) > 1 else "Unknown"
    
    print(f"Plant: {plant}")
    print(f"Disease: {disease}")
    
    # Test the healthy detection logic
    is_healthy_original = 'healthy' in disease.lower()
    is_healthy_class_name = 'healthy' in class_name.lower()
    is_healthy_combined = 'healthy' in class_name.lower() or 'healthy' in disease.lower()
    
    print(f"'healthy' in disease.lower(): {is_healthy_original}")
    print(f"'healthy' in class_name.lower(): {is_healthy_class_name}")
    print(f"Combined logic: {is_healthy_combined}")
    
    return plant, disease, is_healthy_combined

# Test with some examples
print("Testing healthy detection logic...")

# Test healthy plants
healthy_examples = [
    "Apple___healthy",
    "Tomato___healthy", 
    "Potato___healthy",
    "Grape___healthy"
]

# Test diseased plants
diseased_examples = [
    "Apple___Apple_scab",
    "Tomato___Late_blight",
    "Potato___Early_blight",
    "Grape___Black_rot"
]

print("\n" + "="*50)
print("HEALTHY PLANTS (should show True)")
print("="*50)

for class_name in healthy_examples:
    if class_name in class_names:
        plant, disease, is_healthy = test_get_plant_info(class_name)
        print(f"RESULT: {'✅ CORRECT' if is_healthy else '❌ WRONG'}")

print("\n" + "="*50)
print("DISEASED PLANTS (should show False)")
print("="*50)

for class_name in diseased_examples:
    if class_name in class_names:
        plant, disease, is_healthy = test_get_plant_info(class_name)
        print(f"RESULT: {'✅ CORRECT' if not is_healthy else '❌ WRONG'}")

print("\n" + "="*50)
print("ALL CLASS NAMES AND THEIR DETECTION")
print("="*50)

healthy_count = 0
diseased_count = 0

for class_name in class_names:
    parts = class_name.split('___')
    disease = parts[1].replace('_', ' ').title() if len(parts) > 1 else "Unknown"
    is_healthy = 'healthy' in class_name.lower() or 'healthy' in disease.lower()
    
    status = "HEALTHY" if is_healthy else "DISEASED"
    if is_healthy:
        healthy_count += 1
    else:
        diseased_count += 1
    
    print(f"{class_name:<50} -> {status}")

print(f"\nSUMMARY:")
print(f"Total classes: {len(class_names)}")
print(f"Healthy classes: {healthy_count}")
print(f"Diseased classes: {diseased_count}")