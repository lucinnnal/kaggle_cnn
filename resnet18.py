# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import wandb
from utils import calculate_class_weights
import torch.nn.functional as F
from datetime import datetime

# This is for the progress bar.
from tqdm.auto import tqdm
import random

# Set experiment name and wandb project
_exp_name = "food_classification_improved"
project_name = "food_classification"

"""
# Set a random seed for reproducibility
myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
random.seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
"""

# Enhanced data transformations for training with various augmentations
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (224x224)
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_ops=10, magnitude=10),  # Add RandAugment
    # Convert to tensor
    transforms.ToTensor(),
    # Normalize the image
    transforms.Normalize(mean=[0.5555, 0.4514, 0.3443], std=[0.2701, 0.2729, 0.2792]),
])

# Test/validation transformations
test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5555, 0.4514, 0.3443], std=[0.2701, 0.2729, 0.2792]),
])

class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files=None, is_test=False):
        super(FoodDataset).__init__()
        self.path = path
        self.is_test = is_test
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files is not None:
            self.files = files
        print(f"One {path} sample", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname).convert('RGB')  # Ensure RGB format
        im = self.transform(im)
        file_id = os.path.basename(fname).split(".")[0]  # Extract the ID from the filename
        if self.is_test:
            return im, -1, file_id  # Return file_id only for test set
        else:
            label = int(fname.split("/")[-1].split("_")[0])
            return im, label

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=11):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        # Initial convolution layer for 224x224 input
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 56x56
        )

        # ResNet stages
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)   # 56x56
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  # 28x28
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  # 14x14
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)  # 7x7

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 * BasicBlock.expansion, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)      # [B, 64, 56, 56]
        x = self.layer1(x)     # [B, 64, 56, 56]
        x = self.layer2(x)     # [B, 128, 28, 28]
        x = self.layer3(x)     # [B, 256, 14, 14]
        x = self.layer4(x)     # [B, 512, 7, 7]
        
        x = self.avgpool(x)    # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    train_loss = []
    train_accs = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(pbar):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc.item())
        
        # Print every 50 steps
        if (step + 1) % 50 == 0:
            current_loss = sum(train_loss[-50:]) / 50
            current_acc = sum(train_accs[-50:]) / 50
            print(f"\nStep {step+1}/{len(train_loader)}")
            print(f"Loss: {current_loss:.4f}, Accuracy: {current_acc:.4f}")
        
        # Update progress bar
        pbar.set_postfix({
            'loss': sum(train_loss[-10:]) / min(len(train_loss), 10),
            'acc': sum(train_accs[-10:]) / min(len(train_accs), 10)
        })
    
    scheduler.step()
    return np.mean(train_loss), np.mean(train_accs)

def valid_epoch(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = []
    valid_accs = []
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validation"):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc.item())
    
    return np.mean(valid_loss), np.mean(valid_accs)

def main():
    # Initialize wandb
    wandb.init(project=project_name, name=_exp_name, config={
        "learning_rate": 0.0005,
        "epochs": 500,
        "batch_size": 64,
        "model": "ResNet18-300",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingWarmRestarts",
        "scheduler_T0": 50,
        "scheduler_T_mult": 2,
        "scheduler_eta_min": 1e-8,
        "weight_decay": 1e-5,
        "label_smoothing": 0.1
    })
    
    # Hyperparameters
    batch_size = 64
    n_epochs = 500
    patience = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset directory
    _dataset_dir = "/content/data/"
    
    # Load datasets
    train_set = FoodDataset(os.path.join(_dataset_dir, "train"), tfm=train_tfm)
    valid_set = FoodDataset(os.path.join(_dataset_dir, "validation"), tfm=test_tfm)
    
    # Initial training dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Calculate class weights
    class_weights = calculate_class_weights(os.path.join(_dataset_dir, "train")).to(device)
    
    # Initialize model, criterion with class weights, optimizer, scheduler
    model = ResNet18(num_classes=11).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # Phase 1: Initial scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=50,           # First restart at epoch 30
        T_mult=2,         # Keep same cycle length
        eta_min=1e-8,     # Minimum learning rate
        last_epoch=-1
    )
    
    # Update wandb config to include class weights information
    wandb.config.update({
        "class_weights_enabled": True,
        "class_weights": class_weights.cpu().tolist()
    })
    
    # First phase: Train with validation
    print("Phase 1: Training with validation...")
    best_acc = 0
    best_state = None
    stale = 0
    
    for epoch in range(n_epochs):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        
        # Validation
        valid_loss, valid_acc = valid_epoch(model, valid_loader, criterion, device)
        
        # Logging
        wandb.log({
            "phase1/train_loss": train_loss,
            "phase1/train_acc": train_acc,
            "phase1/valid_loss": valid_loss,
            "phase1/valid_acc": valid_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1
        })
        
        # Save best model and check early stopping
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_state = model.state_dict().copy()
            # Save checkpoint for phase 1 with timestamp and accuracy
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"{_exp_name}_phase1_{timestamp}_acc{best_acc:.4f}.ckpt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_acc': best_acc,
            }, save_path)
            print(f"Best model saved at epoch {epoch+1} with accuracy: {best_acc:.4f}")
            # Keep track of best checkpoint path
            best_checkpoint_path = save_path
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Second phase: Train on full dataset with fewer epochs
    print("\nPhase 2: Training on full dataset...")
    n_epochs_phase2 = 100  # Reduced epochs for phase 2
    
    # Combine datasets and create new dataloader
    full_dataset = ConcatDataset([train_set, valid_set])
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Initialize model with best weights from phase 1
    model = ResNet18(num_classes=11).to(device)
    model.load_state_dict(best_state)
    
    # Reset optimizer and scheduler for phase 2 with adjusted epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=100,           # First restart at epoch 20
        T_mult=2,         # Keep same cycle length
        eta_min=1e-8,     # Minimum learning rate
        last_epoch=-1
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # Use same weights
    
    # Train on full dataset
    best_acc_phase2 = 0
    stale = 0
    
    for epoch in range(n_epochs_phase2):
        model.train()
        loss, acc = train_epoch(model, full_loader, criterion, optimizer, scheduler, device, epoch)
        
        # Logging
        wandb.log({
            "phase2/train_loss": loss,
            "phase2/train_acc": acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1
        })
        
        # Save best model and check early stopping for phase 2
        if acc > best_acc_phase2:
            best_acc_phase2 = acc
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"{_exp_name}_phase2_{timestamp}_acc{acc:.4f}.ckpt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'final_acc': acc
            }, save_path)
            print(f"Best model saved at epoch {epoch+1} with accuracy: {acc:.4f}")
            # Keep track of final best checkpoint path
            final_checkpoint_path = save_path
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs in phase 2")
                break
    
    wandb.finish()
    return model

def test_prediction():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _dataset_dir = "/content/data/"
    batch_size = 64
    
    # Load test dataset
    test_set = FoodDataset(os.path.join(_dataset_dir, "test"), tfm=test_tfm, is_test=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load final model
    model = ResNet18(num_classes=11).to(device)
    checkpoint = torch.load(f"{_exp_name}_final.ckpt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate predictions
    predictions = []
    file_ids = []
    
    with torch.no_grad():
        for data, _, file_id in tqdm(test_loader, desc="Generating predictions"):
            outputs = model(data.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            file_ids.extend(file_id)
    
    # Create submission file
    df = pd.DataFrame({
        "ID": file_ids,
        "Category": predictions
    })
    df.to_csv("submission.csv", index=False)
    print("Submission file created successfully!")


if __name__ == "__main__":
    main()