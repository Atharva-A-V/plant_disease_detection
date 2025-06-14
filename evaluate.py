#!/usr/bin/env python3
"""
Plant Disease Detection - Evaluation Script
===========================================

This script evaluates a trained model and provides detailed metrics.

Usage:
    python evaluate.py --model_path best_plant_disease_model.pth --data_path ./PlantVillage-Dataset/raw/color
"""

import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from PIL import Image

class PlantDiseaseEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
    def load_class_names(self):
        """Load class names from file or dataset"""
        if os.path.exists('class_names.json'):
            with open('class_names.json', 'r') as f:
                self.class_names = json.load(f)
        else:
            # Load from dataset
            valid_dir = os.path.join(self.args.data_path, "valid")
            if os.path.exists(valid_dir):
                dataset = datasets.ImageFolder(root=valid_dir)
                self.class_names = dataset.classes
            else:
                raise FileNotFoundError("Class names not found. Please ensure class_names.json exists or valid dataset is available.")
        
        self.num_classes = len(self.class_names)
        print(f"📋 Loaded {self.num_classes} classes")
        
    def load_model(self):
        """Load the trained model"""
        print("🧠 Loading model...")
        
        # Create model architecture
        self.model = models.efficientnet_b3(weights=None)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, self.num_classes)
        )
        
        # Load weights
        if os.path.exists(self.args.model_path):
            state_dict = torch.load(self.args.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Model loaded from {self.args.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {self.args.model_path}")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def create_data_loader(self):
        """Create validation data loader"""
        print("🔄 Creating data loader...")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        valid_dir = os.path.join(self.args.data_path, "valid")
        valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
        
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=32, shuffle=False, num_workers=4
        )
        
        print(f"📊 Validation samples: {len(valid_dataset)}")
        
    def evaluate(self):
        """Evaluate the model"""
        print("🔍 Evaluating model...")
        
        correct, total = 0, 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.valid_loader, desc="Evaluating", colour='cyan'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
        accuracy = 100.0 * correct / total
        print(f"\n🎯 Overall Accuracy: {accuracy:.2f}%")
        
        return all_labels, all_predictions, all_probabilities, accuracy
        
    def generate_reports(self, labels, predictions, probabilities, accuracy):
        """Generate detailed evaluation reports"""
        print("\n📋 Generating detailed reports...")
        
        # Classification report
        report = classification_report(
            labels, predictions, target_names=self.class_names, output_dict=True
        )
        
        print("\n" + "="*80)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(labels, predictions, target_names=self.class_names))
        
        # Per-class accuracy
        print("\n📊 Per-Class Metrics:")
        print("-" * 70)
        print(f"{'Class':<35} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)
        
        for i, class_name in enumerate(self.class_names):
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name:<35} {metrics['precision']:.3f}     {metrics['recall']:.3f}     {metrics['f1-score']:.3f}")
        
        # Overall metrics
        print(f"\n📈 Overall Metrics:")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
        print(f"   Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}")
        
        return report
        
    def plot_confusion_matrix(self, labels, predictions):
        """Plot confusion matrix"""
        print("📊 Generating confusion matrix...")
        
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 Confusion matrix saved as 'confusion_matrix.png'")
        
    def test_time_augmentation(self, image, num_augmentations=5):
        """Apply test-time augmentation"""
        predictions = []
        
        # Original prediction
        with torch.no_grad():
            output = self.model(image)
            predictions.append(torch.softmax(output, dim=1))
        
        # Augmented predictions
        for _ in range(num_augmentations):
            # Apply random transformations
            aug_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Convert tensor back to PIL and apply augmentation
            img_tensor = image.squeeze(0).cpu()
            # Denormalize first
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = img_tensor * std + mean
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            aug_img = aug_transform(img_tensor).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(aug_img)
                predictions.append(torch.softmax(output, dim=1))
        
        # Average all predictions
        avg_prediction = torch.mean(torch.stack(predictions), dim=0)
        return avg_prediction
        
    def predict_single_image(self, image_path, use_tta=True):
        """Predict disease for a single image"""
        print(f"🔍 Analyzing image: {image_path}")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        if use_tta:
            prediction = self.test_time_augmentation(image_tensor)
        else:
            with torch.no_grad():
                output = self.model(image_tensor)
                prediction = torch.softmax(output, dim=1)
        
        # Get top predictions
        probabilities = prediction.squeeze().cpu().numpy()
        top_indices = np.argsort(probabilities)[-5:][::-1]
        
        print(f"\n🎯 Top 5 Predictions:")
        print("-" * 60)
        print(f"{'Rank':<6}{'Disease':<35}{'Confidence':<15}")
        print("-" * 60)
        
        for i, idx in enumerate(top_indices, 1):
            disease_name = self.class_names[idx]
            confidence = probabilities[idx] * 100
            print(f"{i:<6}{disease_name:<35}{confidence:.2f}%")
        
        return self.class_names[top_indices[0]], probabilities[top_indices[0]] * 100

def main():
    parser = argparse.ArgumentParser(description='Evaluate Plant Disease Detection Model')
    parser.add_argument('--model_path', type=str, default='best_plant_disease_model.pth',
                      help='Path to the trained model file')
    parser.add_argument('--data_path', type=str, default='./PlantVillage-Dataset/raw/color',
                      help='Path to the dataset directory')
    parser.add_argument('--single_image', type=str, default=None,
                      help='Path to a single image for prediction')
    parser.add_argument('--use_tta', action='store_true', default=True,
                      help='Use test-time augmentation')
    parser.add_argument('--plot_cm', action='store_true', default=True,
                      help='Plot confusion matrix')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = PlantDiseaseEvaluator(args)
    
    # Load class names
    evaluator.load_class_names()
    
    # Load model
    evaluator.load_model()
    
    if args.single_image:
        # Single image prediction
        predicted_class, confidence = evaluator.predict_single_image(
            args.single_image, use_tta=args.use_tta
        )
        print(f"\n🎯 Final Prediction: {predicted_class} ({confidence:.2f}% confidence)")
    else:
        # Full evaluation
        evaluator.create_data_loader()
        labels, predictions, probabilities, accuracy = evaluator.evaluate()
        report = evaluator.generate_reports(labels, predictions, probabilities, accuracy)
        
        if args.plot_cm:
            evaluator.plot_confusion_matrix(labels, predictions)

if __name__ == '__main__':
    main()