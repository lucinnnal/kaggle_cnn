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
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

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
    transforms.Resize((144, 144)),
    # Random crop to 128x128
    transforms.RandomCrop(128),
    # Random horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),
    # Convert to tensor
    transforms.ToTensor()
])

# Test/validation transformations
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
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

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=11):
        super(ResNet, self).__init__()
        # 초기 채널 수를 32로 줄임 (기존 64)
        self.in_planes = 32

        # 초기 컨볼루션 레이어 - 커널 크기와 stride 줄임
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # ResNet 레이어 - 블록 수와 채널 수 감소
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128 * block.expansion, num_classes)
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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        # ResNet blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Global average pooling and FC layer
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# ResNet 생성 함수 수정
def create_model(num_classes=11):
    # 레이어 수를 [1,1,1]로 줄임 (기존 [2,2,2,2])
    return ResNet(BasicBlock, [1, 1, 1], num_classes)


def main():
    # Initialize wandb
    wandb.init(project=project_name, name=_exp_name, config={
        "learning_rate": 0.0003,
        "epochs": 100,
        "batch_size": 64,
        "model": "ResNet18-modified",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "eta_min": 1e-4,
        "T_max": 30,
        "label_smoothing": 0
    })
    
    # Hyperparameters
    batch_size = 64
    n_epochs = 100
    patience = 10  # Number of epochs to wait for improvement
    
    # Dataset directory
    _dataset_dir = "/content/data/"
    
    # Construct datasets
    train_set = FoodDataset(os.path.join(_dataset_dir, "train"), tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_set = FoodDataset(os.path.join(_dataset_dir, "validation"), tfm=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    model = create_model(num_classes=11).to(device)
    
    # Initialize wandb to watch the model
    wandb.watch(model, log="all")
    
    # Loss function (with label smoothing for better generalization)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-4)
    
    # Training tracking variables
    stale = 0
    best_acc = 0
    
    # Training and validation loop
    for epoch in range(n_epochs):
        # ---------- Training ----------
        model.train()
        train_loss = []
        train_accs = []
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
        
        for batch in train_pbar:
            # Get data and labels
            imgs, labels = batch
            
            # Forward pass
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate accuracy
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            
            # Record metrics
            train_loss.append(loss.item())
            train_accs.append(acc.item())
            
            # Update progress bar
            train_pbar.set_postfix(
                {"loss": sum(train_loss) / len(train_loss), "acc": sum(train_accs) / len(train_accs)}
            )
        
        # Calculate epoch metrics
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        
        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1
        })
        
        # Print training info
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        
        # ---------- Validation ----------
        model.eval()
        valid_loss = []
        valid_accs = []
        
        # Progress bar for validation
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Valid]")
        
        with torch.no_grad():
            for batch in valid_pbar:
                imgs, labels = batch
                logits = model(imgs.to(device))
                loss = criterion(logits, labels.to(device))
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
                
                # Record metrics
                valid_loss.append(loss.item())
                valid_accs.append(acc.item())
                
                # Update progress bar
                valid_pbar.set_postfix(
                    {"loss": sum(valid_loss) / len(valid_loss), "acc": sum(valid_accs) / len(valid_accs)}
                )
        
        # Calculate epoch metrics
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        
        # Log metrics to wandb
        wandb.log({
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "epoch": epoch + 1
        })
        
        # Print validation info
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
        # Update learning rate
        scheduler.step()
        
        # Check for improvement
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch + 1}, saving model")
            best_acc = valid_acc
            stale = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, f"{_exp_name}_best.ckpt")
            
            # Log best model to wandb
            wandb.run.summary["best_accuracy"] = best_acc
            wandb.run.summary["best_epoch"] = epoch + 1
            
        else:
            stale += 1
            print(f"No improvement in validation accuracy for {stale} epochs")
            if stale >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    # Finish wandb run
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
    model_best = create_model(num_classes=11).to(device)
    checkpoint = torch.load(f"{_exp_name}_best.ckpt")
    model_best.load_state_dict(checkpoint['model_state_dict'])
    model_best.eval()
    
    # Prediction with tqdm progress bar
    prediction = []
    file_ids = []
    
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