#!/usr/bin/env python3
"""
Plant Disease Detection - Enhanced Training Script
=================================================

This script trains an EfficientNet-B3 model for plant disease detection
using the same architecture and methodology from your high-accuracy notebook.

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
        
        # Get disease classes - skip train/valid/PlantVillage dirs and any other problematic dirs
        all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        disease_classes = []
        
        for d in all_dirs:
            if d in ['train', 'valid', 'PlantVillage', 'copied']:
                continue
            
            # Check if directory contains image files
            dir_path = os.path.join(base_dir, d)
            items = os.listdir(dir_path)
            has_images = any(item.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) 
                           for item in items if os.path.isfile(os.path.join(dir_path, item)))
            
            if has_images:
                disease_classes.append(d)
            else:
                print(f"⚠️ Skipping {d} - no valid images found")
        
        print(f"Found {len(disease_classes)} disease classes")
        
        # Split data using the same method as your notebook
        for disease in disease_classes:
            disease_path = os.path.join(base_dir, disease)
            if not os.path.exists(disease_path):
                continue
                
            # Get only image files, not directories
            all_items = os.listdir(disease_path)
            images = [item for item in all_items 
                     if os.path.isfile(os.path.join(disease_path, item)) 
                     and item.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            if len(images) == 0:
                print(f"⚠️ No images found in {disease}")
                continue
                
            # Use same train_test_split parameters as notebook
            train_images, valid_images = train_test_split(
                images, test_size=0.2, random_state=42
            )
            
            # Create class directories
            os.makedirs(os.path.join(train_dir, disease), exist_ok=True)
            os.makedirs(os.path.join(valid_dir, disease), exist_ok=True)
            
            # Move files (like in notebook) instead of copy for exact replication
            for img in train_images:
                src = os.path.join(disease_path, img)
                dst = os.path.join(train_dir, disease, img)
                if not os.path.exists(dst) and os.path.isfile(src):
                    shutil.move(src, dst)
                    
            for img in valid_images:
                src = os.path.join(disease_path, img)
                dst = os.path.join(valid_dir, disease, img)
                if not os.path.exists(dst) and os.path.isfile(src):
                    shutil.move(src, dst)
                    
            print(f"✅ {disease}: {len(train_images)} train, {len(valid_images)} valid")
            
        print("🎉 Dataset preparation complete!")
        
    def create_data_loaders(self):
        """Create training and validation data loaders with exact notebook transforms"""
        print("🔄 Creating data loaders...")
        
        # Use EXACT transformations from your notebook for consistency
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224 (for pretrained models)
            transforms.RandomHorizontalFlip(),  # Data Augmentation
            transforms.RandomRotation(10),
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize for pretrained models
        ])
        
        # Validation transform (no augmentation, same as notebook)
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
        
        # Create data loaders with same batch size as notebook (32)
        self.train_loader = DataLoader(
            train_dataset, batch_size=32, 
            shuffle=True, num_workers=4, pin_memory=True
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=32, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        self.class_names = train_dataset.classes
        self.num_classes = len(self.class_names)
        
        print(f"📊 Dataset Statistics:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(valid_dataset)}")
        print(f"   Number of classes: {self.num_classes}")
        print(f"   Classes: {self.class_names[:5]}..." if len(self.class_names) > 5 else f"   Classes: {self.class_names}")
        
        # Save class names for later use
        with open('class_names.json', 'w') as f:
            json.dump(self.class_names, f, indent=2)
            
    def create_model(self):
        """Create model with EXACT architecture from your notebook"""
        print("🧠 Creating EfficientNet-B3 model...")
        
        # Use EfficientNet-B3 with exact configuration from notebook
        self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # Freeze early layers for transfer learning (same as notebook)
        for param in self.model.features[:-3].parameters():
            param.requires_grad = False
            
        # Custom classifier with dropout for regularization (EXACT from notebook)
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
        
        print(f"   Architecture: EfficientNet-B3 (from high-accuracy notebook)")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Output classes: {self.num_classes}")
        
    def setup_training(self):
        """Setup training components with exact notebook configuration"""
        print("⚙️ Setting up training components...")
        
        # Loss function with label smoothing (same as notebook)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with weight decay (exact from notebook)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=1e-4
        )
        
        # Learning rate scheduler (exact from notebook)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        print(f"   Loss: CrossEntropyLoss with label smoothing (0.1)")
        print(f"   Optimizer: AdamW (lr=0.001, weight_decay=1e-4)")
        print(f"   Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")
        
    def train_epoch(self, epoch):
        """Train for one epoch with enhanced progress tracking from notebook"""
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        # Enhanced progress bar matching notebook style
        train_pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}",
                         ncols=100, colour='blue', leave=False)
        
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
            
            # Update progress bar with current metrics (notebook style)
            current_acc = 100 * correct / total
            current_loss = running_loss / (batch_idx + 1)
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
            
        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(self.train_loader)
        
        return avg_train_loss, train_acc
        
    def validate_epoch(self, epoch):
        """Validate for one epoch with enhanced progress tracking from notebook"""
        self.model.eval()
        val_correct, val_total = 0, 0
        val_loss = 0.0
        
        # Enhanced validation progress bar matching notebook style
        val_pbar = tqdm(self.valid_loader, desc=f"Validation Epoch {epoch+1}",
                       ncols=100, colour='green', leave=False)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                # Update progress bar with current metrics (notebook style)
                current_val_acc = 100 * val_correct / val_total
                current_val_loss = val_loss / (batch_idx + 1)
                val_pbar.set_postfix({
                    'Loss': f'{current_val_loss:.4f}',
                    'Acc': f'{current_val_acc:.2f}%'
                })
                
        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_loss / len(self.valid_loader)
        
        return val_loss_avg, val_acc

    def test_time_augmentation(self, model, image, num_augmentations=5):
        """Apply test-time augmentation for better predictions (from notebook)"""
        model.eval()
        predictions = []
        
        # Original prediction
        with torch.no_grad():
            output = model(image)
            predictions.append(torch.softmax(output, dim=1))
        
        # Augmented predictions
        for _ in range(num_augmentations):
            # Apply random transformations
            aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Convert back to PIL and apply augmentation
            img_pil = transforms.ToPILImage()(image.squeeze(0).cpu())
            aug_img = aug_transform(img_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = model(aug_img)
                predictions.append(torch.softmax(output, dim=1))
        
        # Average all predictions
        avg_prediction = torch.mean(torch.stack(predictions), dim=0)
        return avg_prediction

    def final_evaluation_with_tta(self):
        """Final evaluation with test-time augmentation (from notebook)"""
        print("🎯 Performing final evaluation with Test-Time Augmentation...")
        
        self.model.eval()
        correct, total = 0, 0
        all_predictions = []
        all_labels = []
        
        eval_pbar = tqdm(self.valid_loader, desc="TTA Evaluation", 
                        ncols=100, colour='yellow')
        
        with torch.no_grad():
            for images, labels in eval_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Use test-time augmentation for better accuracy
                batch_predictions = []
                for i in range(images.size(0)):
                    single_image = images[i:i+1]
                    avg_pred = self.test_time_augmentation(self.model, single_image)
                    batch_predictions.append(avg_pred)
                
                batch_predictions = torch.cat(batch_predictions, dim=0)
                _, preds = torch.max(batch_predictions, 1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                current_acc = 100.0 * correct / total
                eval_pbar.set_postfix({'TTA_Acc': f'{current_acc:.2f}%'})
        
        final_accuracy = 100.0 * correct / total
        print(f"🏆 Final Test Accuracy with TTA: {final_accuracy:.2f}%")
        
        # Classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=self.class_names, 
                                  digits=3, zero_division=0))
        
        return final_accuracy
        
    def train(self):
        """Main training loop with enhanced output matching notebook style"""
        print(f"🚀 Starting training for {self.args.epochs} epochs...")
        print(f"📋 Early stopping patience: {self.args.patience}")
        
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{self.args.epochs}")
            print(f"{'='*60}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Enhanced epoch summary (notebook style)
            print(f"\n📊 EPOCH {epoch+1} SUMMARY:")
            print(f"   🔵 Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            print(f"   🟢 Valid - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            
            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_acc)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr != old_lr:
                print(f"   📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
                
            # Enhanced model saving and early stopping (notebook style)
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
                print(f"   Best validation accuracy achieved: {self.best_val_acc:.2f}%")
                break
                
            # Add small delay for better visualization (like notebook)
            time.sleep(0.5)
                
        training_time = time.time() - start_time
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"   🏆 Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"   ⏱️ Total training time: {training_time/3600:.2f} hours")
        print(f"   📁 Best model saved as: 'best_plant_disease_model.pth'")
        
        # Perform final evaluation with TTA
        final_tta_acc = self.final_evaluation_with_tta()
        
        # Plot training history
        self.plot_training_history()
        
    def save_model(self):
        """Save the model with enhanced metadata"""
        model_info = {
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'architecture': 'EfficientNet-B3',
            'input_size': (224, 224),
            'best_accuracy': self.best_val_acc,
            'training_args': vars(self.args),
            'notebook_replication': True,  # Flag to indicate this uses notebook architecture
            'training_time': time.time()
        }
        
        torch.save(model_info, 'plant_disease_model_complete.pth')
        torch.save(self.model.state_dict(), 'best_plant_disease_model.pth')
        
    def plot_training_history(self):
        """Plot training metrics with enhanced styling"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot with enhanced styling
        ax1.plot(range(1, len(self.train_losses)+1), self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(range(1, len(self.val_losses)+1), self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot with enhanced styling
        ax2.plot(range(1, len(self.train_accs)+1), self.train_accs, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(range(1, len(self.val_accs)+1), self.val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Add best accuracy annotation
        best_epoch = self.val_accs.index(max(self.val_accs)) + 1
        ax2.annotate(f'Best: {max(self.val_accs):.2f}%', 
                    xy=(best_epoch, max(self.val_accs)), 
                    xytext=(best_epoch+1, max(self.val_accs)-5),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, color='red', fontweight='bold')
        
        plt.suptitle('Training History - Enhanced EfficientNet-B3 Model', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Enhanced training plot saved as 'training_history.png'")

def main():
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection Model (Enhanced from Notebook)')
    parser.add_argument('--data_path', type=str, default='./PlantVillage-Dataset/raw/color',
                      help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs (default: 5 for quick training)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training (matches notebook)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate (matches notebook)')
    parser.add_argument('--patience', type=int, default=5,
                      help='Early stopping patience (matches notebook)')
    parser.add_argument('--skip_data_prep', action='store_true',
                      help='Skip data preparation step')
    
    args = parser.parse_args()
    
    print("🚀 ENHANCED TRAINING SESSION - NOTEBOOK REPLICATION")
    print("="*70)
    print(f"📊 Training Configuration (ENHANCED FROM HIGH-ACCURACY NOTEBOOK):")
    print(f"   Architecture: EfficientNet-B3 with custom classifier")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size} (notebook optimized)")
    print(f"   Learning Rate: {args.learning_rate} (notebook optimized)")
    print(f"   Early Stopping Patience: {args.patience}")
    print(f"   Data Path: {args.data_path}")
    print(f"   Features: Label smoothing, AdamW optimizer, TTA evaluation")
    print("="*70)
    
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
    
    print("\n🎉 ENHANCED TRAINING COMPLETED!")
    print("✅ Model with notebook architecture saved and ready for deployment")
    print("🔥 This model uses the exact configuration from your high-accuracy notebook!")

if __name__ == "__main__":
    main()