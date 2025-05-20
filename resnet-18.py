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
from sklearn.model_selection import StratifiedKFold

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
        "scheduler_milestones": [30, 50, 70, 90],
        "scheduler_gamma": 0.7,
        "weight_decay": 1e-5,
        "label_smoothing": 0.1,
        "n_folds": 5,
        "pseudo_threshold": 0.9  # Confidence threshold for pseudo labeling
    })
    
    # Hyperparameters
    batch_size = 64
    n_epochs = 200
    n_folds = 5
    patience = 20
    pseudo_threshold = 0.9
    
    # Dataset directory
    _dataset_dir = "/content/data/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load all data
    train_set = FoodDataset(os.path.join(_dataset_dir, "train"), tfm=train_tfm)
    valid_set = FoodDataset(os.path.join(_dataset_dir, "validation"), tfm=train_tfm)
    test_set = FoodDataset(os.path.join(_dataset_dir, "test"), tfm=test_tfm, is_test=True)
    
    # Combine train and validation sets
    all_data = ConcatDataset([train_set, valid_set])
    
    # Prepare labels for stratification
    all_labels = []
    for dataset in [train_set, valid_set]:
        all_labels.extend([dataset.files[i].split("/")[-1].split("_")[0] for i in range(len(dataset))])
    all_labels = np.array(all_labels, dtype=int)
    
    # Initialize fold predictions for test set
    test_predictions = []
    
    # 5-fold cross validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=myseed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\nFold {fold + 1}")
        
        # Create fold datasets
        train_fold = Subset(all_data, train_idx)
        val_fold = Subset(all_data, val_idx)
        
        train_loader = DataLoader(train_fold, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(val_fold, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Initialize model, criterion, optimizer
        model = ResNet18(num_classes=11).to(device)
        class_weights = calculate_class_weights(os.path.join(_dataset_dir, "train")).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=1e-5)
        scheduler = MultiStepLR(optimizer, milestones=[30, 50, 70, 90], gamma=0.7)
        
        # Training tracking variables
        best_acc = 0
        stale = 0
        
        # Training loop
        for epoch in range(n_epochs):
            # Training
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
            
            # Validation
            valid_loss, valid_acc = valid_epoch(model, valid_loader, criterion, device)
            
            # Logging
            wandb.log({
                f"fold_{fold+1}/train_loss": train_loss,
                f"fold_{fold+1}/train_acc": train_acc,
                f"fold_{fold+1}/valid_loss": valid_loss,
                f"fold_{fold+1}/valid_acc": valid_acc,
                "epoch": epoch + 1
            })
            
            # Save best model
            if valid_acc > best_acc:
                best_acc = valid_acc
                stale = 0
                torch.save({
                    'fold': fold + 1,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'best_acc': best_acc,
                }, f"{_exp_name}_fold{fold+1}_best.ckpt")
            else:
                stale += 1
                if stale >= patience:
                    break
        
        # Generate predictions for test set
        model.eval()
        fold_predictions = []
        fold_probabilities = []
        
        with torch.no_grad():
            for data, _, _ in test_loader:
                outputs = model(data.to(device))
                probs = torch.softmax(outputs, dim=1)
                fold_probabilities.append(probs.cpu())
                
        fold_probabilities = torch.cat(fold_probabilities)
        test_predictions.append(fold_probabilities)
    
    # Average predictions from all folds
    ensemble_probs = torch.stack(test_predictions).mean(0)
    ensemble_preds = ensemble_probs.argmax(1)
    
    # Generate pseudo labels for high confidence predictions
    pseudo_mask = ensemble_probs.max(1)[0] >= pseudo_threshold
    pseudo_labels = ensemble_preds[pseudo_mask]
    pseudo_data = Subset(test_set, pseudo_mask.nonzero().squeeze())
    
    # Train final model with pseudo labels
    print("\nTraining final model with pseudo labels...")
    
    # Combine all data with pseudo labeled data
    final_train_data = ConcatDataset([all_data, pseudo_data])
    final_train_loader = DataLoader(final_train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Train final model
    final_model = ResNet18(num_classes=11).to(device)
    final_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=0.0004, weight_decay=1e-5)
    final_scheduler = MultiStepLR(final_optimizer, milestones=[30, 50, 70, 90], gamma=0.7)
    
    # Final training loop
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(final_model, final_train_loader, final_criterion, 
                                          final_optimizer, final_scheduler, device, epoch)
        
        wandb.log({
            "final_model/train_loss": train_loss,
            "final_model/train_acc": train_acc,
            "epoch": epoch + 1
        })
    
    # Save final model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': final_model.state_dict(),
        'final_acc': train_acc,
    }, f"{_exp_name}_final.ckpt")
    
    # Generate final predictions
    final_predictions = generate_predictions(final_model, test_loader, device)
    
    # Create submission CSV
    create_submission(final_predictions, test_set.files)
    
    wandb.finish()

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    train_loss = []
    train_accs = []
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch in train_pbar:
        imgs, labels = batch
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        
        train_loss.append(loss.item())
        train_accs.append(acc.item())
        
        train_pbar.set_postfix(
            {"loss": sum(train_loss) / len(train_loss), "acc": sum(train_accs) / len(train_accs)}
        )
    
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    
    return train_loss, train_acc

def valid_epoch(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = []
    valid_accs = []
    
    valid_pbar = tqdm(valid_loader, desc="[Valid]")
    
    with torch.no_grad():
        for batch in valid_pbar:
            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            
            valid_loss.append(loss.item())
            valid_accs.append(acc.item())
            
            valid_pbar.set_postfix(
                {"loss": sum(valid_loss) / len(valid_loss), "acc": sum(valid_accs) / len(valid_accs)}
            )
    
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    
    return valid_loss, valid_acc

def generate_predictions(model, loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for data, _, _ in tqdm(loader, desc="Generating predictions"):
            outputs = model(data.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
    
    return predictions

def create_submission(predictions, files):
    file_ids = [os.path.basename(f).split(".")[0] for f in files]
    df = pd.DataFrame({"ID": file_ids, "Category": predictions})
    df.to_csv("submission.csv", index=False)
    print("Submission file created successfully!")

if __name__ == "__main__":
    main()