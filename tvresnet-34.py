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

# Enhanced data transformations
train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
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

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, num_classes=11, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) 
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
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
        "scheduler_milestones": [60, 120, 160],  # Milestones for lr decay
        "scheduler_gamma": 0.1,  # Learning rate decay factor
        "weight_decay": 1e-5,
        "label_smoothing": 0.1
    })
    
    # Hyperparameters
    batch_size = 64
    n_epochs = 200
    patience = 20  # Number of epochs to wait for improvement
    
    # Dataset directory
    _dataset_dir = "/content/data/"
    
    # Calculate class weights from training set
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights = calculate_class_weights(os.path.join(_dataset_dir, "train")).to(device)
    
    # Construct datasets
    train_set = FoodDataset(os.path.join(_dataset_dir, "train"), tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_set = FoodDataset(os.path.join(_dataset_dir, "validation"), tfm=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model - Tiny version configuration
    model = ConvNeXt(
        num_classes=11,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1
    ).to(device)
    
    # Initialize wandb to watch the model
    wandb.watch(model, log="all")
    
    # Loss function (with label smoothing for better generalization)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )
    
    # Update wandb config to include class weights information
    wandb.config.update({
        "class_weights_enabled": True,
        "class_weights": class_weights.cpu().tolist()
    })
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=1e-5)
    
    # Change learning rate scheduler to MultiStepLR
    scheduler = MultiStepLR(
        optimizer,
        milestones=[30, 50, 70, 90],  # Decrease lr at epochs 60, 120, and 160
        gamma=0.7  # Multiply lr by 0.1 at each milestone
    )
    
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
    model_best = ConvNeXt(
        num_classes=11,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1
    ).to(device)
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