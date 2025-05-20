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
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import wandb
from utils import calculate_class_weights

# This is for the progress bar.
from tqdm.auto import tqdm
import random

# Set experiment name and wandb project
_exp_name = "food_classification_improved"
project_name = "food_classification"

# Set a random seed for reproducibility
myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
random.seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# Enhanced data transformations for training with various augmentations
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape
    transforms.Resize((128, 128)),
    # Random horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),
    # Rotation
    transforms.RandomRotation(30),
    # Color Jitter
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # Convert to tensor
    transforms.ToTensor(),
    # Normalize the image
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

# Test/validation transformations
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=11):
        super(ResNet18, self).__init__()
        self.in_planes = 32  # 채널 수를 줄임

        # Initial convolution layer for 128x128 input
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),  # [32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # [32, 64, 64]
        )

        # ResNet stages with BasicBlock - [2,2,2,2] configuration
        self.layer1 = self._make_layer(BasicBlock, 32, 2, stride=1)    # [32, 64, 64]
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2)    # [64, 32, 32]
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2)   # [128, 16, 16]
        self.layer4 = self._make_layer(BasicBlock, 256, 2, stride=2)   # [256, 8, 8]

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * BasicBlock.expansion, num_classes)
        )

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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)      # [B, 32, 64, 64]
        
        x = self.layer1(x)     # [B, 32, 64, 64]
        x = self.layer2(x)     # [B, 64, 32, 32]
        x = self.layer3(x)     # [B, 128, 16, 16]
        x = self.layer4(x)     # [B, 256, 8, 8]
        
        x = self.avgpool(x)    # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def main():
    # Initialize wandb
    wandb.init(project=project_name, name=_exp_name, config={
        "learning_rate": 0.0003,
        "epochs": 200,
        "batch_size": 64,
        "model": "ResNet18",
        "optimizer": "AdamW",
        "scheduler": "MultiStepLR",
        "scheduler_milestones": [60, 120, 160],
        "scheduler_gamma": 0.1,
        "weight_decay": 1e-5,
        "label_smoothing": 0.1,
        "training_mode": "combined_train_valid"  # 새로운 설정 추가
    })
    
    # Hyperparameters
    batch_size = 64
    n_epochs = 200
    
    # Dataset directory
    _dataset_dir = "/content/data/"
    
    # Calculate class weights using combined dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Combine train and validation datasets
    train_set = FoodDataset(os.path.join(_dataset_dir, "train"), tfm=train_tfm)
    valid_set = FoodDataset(os.path.join(_dataset_dir, "validation"), tfm=train_tfm)  # validation도 train transform 사용
    combined_dataset = ConcatDataset([train_set, valid_set])
    
    # Calculate class weights from combined dataset
    class_weights = calculate_class_weights(os.path.join(_dataset_dir, "train")).to(device)
    
    # Create combined dataloader
    combined_loader = DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = ResNet18(num_classes=11).to(device)
    wandb.watch(model, log="all")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=1e-5)
    scheduler = MultiStepLR(
        optimizer,
        milestones=[30, 50, 70, 90],
        gamma=0.7
    )
    
    # Training loop
    best_loss = float('inf')
    _exp_name = "food_classification_improved_combined"
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = []
        train_accs = []
        
        # Progress bar
        pbar = tqdm(combined_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for step, batch in enumerate(pbar):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Forward pass
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            
            # Record metrics
            train_loss.append(loss.item())
            train_accs.append(acc.item())
            
            # Log every 10 steps
            if (step + 1) % 10 == 0:
                current_loss = sum(train_loss[-10:]) / 10
                current_acc = sum(train_accs[-10:]) / 10
                
                wandb.log({
                    "step": epoch * len(combined_loader) + step,
                    "step_loss": current_loss,
                    "step_acc": current_acc,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                
                pbar.set_postfix({
                    "loss": current_loss,
                    "acc": current_acc
                })
        
        # Calculate epoch metrics
        epoch_loss = sum(train_loss) / len(train_loss)
        epoch_acc = sum(train_accs) / len(train_accs)
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "epoch_loss": epoch_loss,
            "epoch_acc": epoch_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f"[ Epoch {epoch + 1:03d}/{n_epochs:03d} ] loss = {epoch_loss:.5f}, acc = {epoch_acc:.5f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"Best model found at epoch {epoch + 1}, saving model")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, f"{_exp_name}_best.ckpt")
            
            wandb.run.summary["best_loss"] = best_loss
            wandb.run.summary["best_epoch"] = epoch + 1
    
    wandb.finish()
    
    # Test prediction
    test_prediction()
    

def test_prediction():
    _dataset_dir = "/content/data/"
    _exp_name = "food_classification_improved"
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load test dataset
    test_set = FoodDataset(os.path.join(_dataset_dir, "test"), tfm=test_tfm, is_test=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load best model
    model_best = ResNet18(num_classes=11).to(device)
    checkpoint = torch.load(f"{_exp_name}_best.ckpt")
    model_best.load_state_dict(checkpoint['model_state_dict'])
    model_best.eval()
    
    # Prediction with tqdm progress bar
    prediction = []
    file_ids = []3
    
    with torch.no_grad():
        for data, _, file_id in tqdm(test_loader, desc="Generating predictions"):
            test_pred = model_best(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()
            file_ids += file_id  # Append original IDs
    
    # Create submission CSV
    df = pd.DataFrame()
    df["ID"] = file_ids
    df["Category"] = prediction
    df.to_csv("submission.csv", index=False)
    print("Submission file created successfully!")


if __name__ == "__main__":
    main()