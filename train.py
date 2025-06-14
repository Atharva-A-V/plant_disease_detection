#!/usr/bin/env python3
"""
Plant Disease Detection - Training Script
==========================================

This script trains an EfficientNet-B3 model for plant disease detection
using the PlantVillage dataset.

Usage:
    python train.py --data_path ./PlantVillage-Dataset/raw/color --epochs 20 --batch_size 32
"""

import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import time

class PlantDiseaseTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        # Training configuration
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_losses, self.train_accs = [], []
        self.val_losses, self.val_accs = [], []
        
    def prepare_data(self):
        """Split dataset into train/validation sets"""
        print("📂 Preparing dataset...")
        
        base_dir = self.args.data_path
        train_dir = os.path.join(base_dir, "train")
        valid_dir = os.path.join(base_dir, "valid")
        
        # Create directories
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        
        # Get disease classes
        disease_classes = [d for d in os.listdir(base_dir)
                          if os.path.isdir(os.path.join(base_dir, d)) and d not in ['train', 'valid']]
        
        print(f"Found {len(disease_classes)} disease classes")
        
        # Split data
        for disease in disease_classes:
            disease_path = os.path.join(base_dir, disease)
            if not os.path.exists(disease_path):
                continue
                
            images = os.listdir(disease_path)
            if len(images) == 0:
                continue
                
            train_images, valid_images = train_test_split(
                images, test_size=0.2, random_state=42
            )
            
            # Create class directories
            os.makedirs(os.path.join(train_dir, disease), exist_ok=True)
            os.makedirs(os.path.join(valid_dir, disease), exist_ok=True)
            
            # Move images
            for img in train_images:
                src = os.path.join(disease_path, img)
                dst = os.path.join(train_dir, disease, img)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    
            for img in valid_images:
                src = os.path.join(disease_path, img)
                dst = os.path.join(valid_dir, disease, img)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    
            print(f"✅ {disease}: {len(train_images)} train, {len(valid_images)} valid")
            
        print("🎉 Dataset preparation complete!")
        
    def create_data_loaders(self):
        """Create training and validation data loaders"""
        print("🔄 Creating data loaders...")
        
        # Enhanced transformations with data augmentation
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Validation transform (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dir = os.path.join(self.args.data_path, "train")
        valid_dir = os.path.join(self.args.data_path, "valid")
        
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        valid_dataset = datasets.ImageFolder(root=valid_dir, transform=val_transform)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, 
            shuffle=True, num_workers=4, pin_memory=True
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=self.args.batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        self.class_names = train_dataset.classes
        self.num_classes = len(self.class_names)
        
        print(f"📊 Dataset Statistics:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(valid_dataset)}")
        print(f"   Number of classes: {self.num_classes}")
        
        # Save class names for later use
        with open('class_names.json', 'w') as f:
            json.dump(self.class_names, f)
            
    def create_model(self):
        """Create and configure the model"""
        print("🧠 Creating model...")
        
        # Use EfficientNet-B3
        self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # Freeze early layers for transfer learning
        for param in self.model.features[:-3].parameters():
            param.requires_grad = False
            
        # Custom classifier with dropout
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, self.num_classes)
        )
        
        self.model = self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   Architecture: EfficientNet-B3")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Output classes: {self.num_classes}")
        
    def setup_training(self):
        """Setup loss function, optimizer, and scheduler"""
        print("⚙️ Setting up training components...")
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        print(f"   Loss: CrossEntropyLoss with label smoothing (0.1)")
        print(f"   Optimizer: AdamW (lr={self.args.learning_rate}, weight_decay=1e-4)")
        print(f"   Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        train_pbar = tqdm(
            self.train_loader, 
            desc=f"🔵 Training Epoch {epoch+1}",
            ncols=120, colour='blue', leave=False
        )
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            current_acc = 100 * correct / total
            current_loss = running_loss / (batch_idx + 1)
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(self.train_loader)
        
        return avg_train_loss, train_acc
        
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        val_correct, val_total = 0, 0
        val_loss = 0.0
        
        val_pbar = tqdm(
            self.valid_loader,
            desc=f"🟢 Validation Epoch {epoch+1}",
            ncols=120, colour='green', leave=False
        )
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                current_val_acc = 100 * val_correct / val_total
                current_val_loss = val_loss / (batch_idx + 1)
                val_pbar.set_postfix({
                    'Loss': f'{current_val_loss:.4f}',
                    'Acc': f'{current_val_acc:.2f}%'
                })
                
        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_loss / len(self.valid_loader)
        
        return val_loss_avg, val_acc
        
    def train(self):
        """Main training loop"""
        print(f"🚀 Starting training for {self.args.epochs} epochs...")
        print(f"📋 Early stopping patience: {self.args.patience}")
        
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1}/{self.args.epochs}")
            print(f"{'='*70}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Print epoch results
            print(f"\n📊 EPOCH {epoch+1} RESULTS:")
            print(f"   🔵 Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            print(f"   🟢 Valid - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            
            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_acc)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr != old_lr:
                print(f"   📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
                
            # Model saving and early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model()
                self.patience_counter = 0
                print(f"   ⭐ NEW BEST MODEL! Validation Accuracy: {val_acc:.2f}%")
                print(f"   💾 Model saved as 'best_plant_disease_model.pth'")
            else:
                self.patience_counter += 1
                print(f"   ⏳ No improvement. Patience: {self.patience_counter}/{self.args.patience}")
                
            if self.patience_counter >= self.args.patience:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"   🏆 Best validation accuracy achieved: {self.best_val_acc:.2f}%")
                break
                
        training_time = time.time() - start_time
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"   🏆 Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"   ⏱️ Total training time: {training_time/3600:.2f} hours")
        print(f"   📁 Best model saved as: 'best_plant_disease_model.pth'")
        
        # Plot training history
        self.plot_training_history()
        
    def save_model(self):
        """Save the model with metadata"""
        model_info = {
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'architecture': 'EfficientNet-B3',
            'input_size': (224, 224),
            'best_accuracy': self.best_val_acc,
            'training_args': vars(self.args)
        }
        
        torch.save(model_info, 'plant_disease_model_complete.pth')
        torch.save(self.model.state_dict(), 'best_plant_disease_model.pth')
        
    def plot_training_history(self):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(range(1, len(self.train_losses)+1), self.train_losses, 'b-', label='Training Loss')
        ax1.plot(range(1, len(self.val_losses)+1), self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Model Loss Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(range(1, len(self.train_accs)+1), self.train_accs, 'b-', label='Training Accuracy')
        ax2.plot(range(1, len(self.val_accs)+1), self.val_accs, 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy Over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Training plot saved as 'training_history.png'")

def main():
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection Model')
    parser.add_argument('--data_path', type=str, default='./PlantVillage-Dataset/raw/color',
                      help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--patience', type=int, default=7,
                      help='Early stopping patience')
    parser.add_argument('--skip_data_prep', action='store_true',
                      help='Skip data preparation step')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PlantDiseaseTrainer(args)
    
    # Prepare data
    if not args.skip_data_prep:
        trainer.prepare_data()
    
    # Create data loaders
    trainer.create_data_loaders()
    
    # Create model
    trainer.create_model()
    
    # Setup training
    trainer.setup_training()
    
    # Train model
    trainer.train()

if __name__ == '__main__':
    main()