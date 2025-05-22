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
_exp_name = "food_classification"
project_name = "food_classification_EfficientNetV2"

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
    # Resize the image into a fixed shape (224x224)
    transforms.Resize((224, 224)),
    # Random horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),
    # Rotation
    transforms.RandomRotation(30),
    # Color Jitter
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, reduction=4):
        super(MBConv, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio

        layers = []
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU()
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ])

        # SE
        se_dim = max(1, in_channels // reduction)
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, se_dim, 1),
            nn.SiLU(),
            nn.Conv2d(se_dim, hidden_dim, 1),
            nn.Sigmoid()
        ])

        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=11):
        super(EfficientNetV2, self).__init__()

        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU()
        )

        # Fused-MBConv and MBConv blocks
        self.block1 = self._make_layer(24, 48, 2, stride=1)  # Fused-MBConv
        self.block2 = self._make_layer(48, 64, 2, stride=2)  # Fused-MBConv
        self.block3 = self._make_layer(64, 128, 3, stride=2)  # MBConv
        self.block4 = self._make_layer(128, 160, 5, stride=2)  # MBConv
        self.block5 = self._make_layer(160, 256, 5, stride=1)  # MBConv

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(MBConv(in_channels, out_channels, expand_ratio=4, stride=stride))
        for _ in range(1, blocks):
            layers.append(MBConv(out_channels, out_channels, expand_ratio=4, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.head(x)
        return x

def test_prediction():
    _dataset_dir = "/content/data/"
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create final_submission directory if it doesn't exist
    os.makedirs("final_submission", exist_ok=True)
    
    # Load test dataset
    test_set = FoodDataset(os.path.join(_dataset_dir, "test"), tfm=test_tfm, is_test=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Get all checkpoint files
    checkpoint_dir = "/content/checkpoints"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    
    # Process each checkpoint
    for ckpt_file in checkpoint_files:
        print(f"\nProcessing checkpoint: {ckpt_file}")
        
        # Load model and checkpoint
        model = ResNet18(num_classes=11).to(device)
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_file)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get epoch number from checkpoint filename
        epoch_num = ckpt_file.split('_')[-1].split('.')[0]
        
        # Prediction with tqdm progress bar
        prediction = []
        file_ids = []
        
        with torch.no_grad():
            for data, _, file_id in tqdm(test_loader, desc="Generating predictions"):
                test_pred = model(data.to(device))
                test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
                prediction += test_label.squeeze().tolist()
                file_ids += file_id
        
        # Create submission CSV
        df = pd.DataFrame()
        df["ID"] = file_ids
        df["Category"] = prediction
        
        # Save to final_submission folder with unique name
        submission_path = os.path.join("final_submission", f"submission_epoch_{epoch_num}.csv")
        df.to_csv(submission_path, index=False)
        print(f"Saved submission file: {submission_path}")