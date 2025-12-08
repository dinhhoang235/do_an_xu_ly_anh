"""
CNN Character Recognizer for License Plate Characters
Deep Learning approach using PyTorch SimpleCNN architecture
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm


class CharacterDataset(Dataset):
    """PyTorch Dataset for character images"""
    
    def __init__(self, dataset_path, transform=None):
        self.data = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        dataset_path = Path(dataset_path)
        
        # Build label mapping
        label_idx = 0
        for char_dir in sorted(dataset_path.iterdir()):
            if not char_dir.is_dir():
                continue
            
            char_label = char_dir.name
            self.label_to_idx[char_label] = label_idx
            self.idx_to_label[label_idx] = char_label
            label_idx += 1
        
        # Load images
        for char_dir in sorted(dataset_path.iterdir()):
            if not char_dir.is_dir():
                continue
            
            char_label = char_dir.name
            label_idx = self.label_to_idx[char_label]
            
            char_images = list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpg"))
            
            for img_path in char_images:
                self.data.append(str(img_path))
                self.labels.append(label_idx)
        
        self.transform = transform or self.default_transform()
    
    def default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        
        img_tensor = self.transform(img)
        
        return img_tensor, label


class SimpleCNN(nn.Module):
    """Simple CNN for character recognition"""
    
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNNRecognizer:
    """Deep Learning approach using CNN"""
    
    def __init__(self, num_classes=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.num_classes = num_classes
        self.is_trained = False
        self.label_map = {}
        
    def train(self, dataset_path, epochs=20, batch_size=32, learning_rate=0.001):
        """Train CNN model"""
        print("\n🔄 [CNN] Training CNN model...")
        
        # Create dataset
        dataset = CharacterDataset(dataset_path)
        self.num_classes = len(dataset.label_to_idx)
        self.label_map = dataset.label_to_idx
        
        print(f"   📊 Dataset size: {len(dataset)} samples")
        print(f"   📝 Classes: {self.num_classes}")
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Create model
        self.model = SimpleCNN(self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            self.model.eval()
            val_acc = 0
            val_loss = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_acc += (predicted == labels).sum().item()
                    val_loss += loss.item()
            
            val_acc = val_acc / len(val_dataset)
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        self.is_trained = True
        print(f"   ✅ CNN training completed (Best Val Acc: {best_val_acc:.4f})")
    
    def predict(self, char_img):
        """Predict character label"""
        if not self.is_trained:
            raise RuntimeError("Model not trained!")
        
        # Preprocess
        if len(char_img.shape) == 3:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        
        char_img_rgb = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)
        char_img_rgb = cv2.cvtColor(char_img_rgb, cv2.COLOR_BGR2RGB)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        from PIL import Image
        img_tensor = transform(Image.fromarray(char_img_rgb)).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            _, pred = torch.max(output, 1)
        
        pred_idx = pred.item()
        
        # Reverse lookup label
        idx_to_label = {v: k for k, v in self.label_map.items()}
        pred_label = idx_to_label[pred_idx]
        
        return pred_label
    
    def predict_batch(self, char_images):
        """Predict batch of characters"""
        if not self.is_trained:
            raise RuntimeError("Model not trained!")
        
        # Preprocess batch
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        from PIL import Image
        batch_tensor = []
        
        for char_img in char_images:
            if len(char_img.shape) == 3:
                char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
            
            char_img_rgb = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)
            char_img_rgb = cv2.cvtColor(char_img_rgb, cv2.COLOR_BGR2RGB)
            
            img_tensor = transform(Image.fromarray(char_img_rgb))
            batch_tensor.append(img_tensor)
        
        batch_tensor = torch.stack(batch_tensor).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch_tensor)
            _, preds = torch.max(output, 1)
        
        pred_indices = preds.cpu().numpy()
        
        # Reverse lookup labels
        idx_to_label = {v: k for k, v in self.label_map.items()}
        pred_labels = [idx_to_label[idx] for idx in pred_indices]
        
        return np.array(pred_labels)
