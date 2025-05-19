import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import random
from sklearn.utils.class_weight import compute_class_weight  # 추가
import wandb

# 재현성을 위한 시드 설정
myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# 향상된 데이터 변환
test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        im = Image.open(fname)
        im = self.transform(im)
        file_id = os.path.basename(fname).split(".")[0]
        if self.is_test:
            return im, -1, file_id
        else:
            label = int(fname.split("/")[-1].split("_")[0])
            return im, label

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class EnhancedClassifier(nn.Module):
    def __init__(self, num_classes=11):
        super(EnhancedClassifier, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    train_loss = []
    train_accs = []
    
    for step, batch in enumerate(tqdm(train_loader)):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        
        mixed_imgs, labels_a, labels_b, lam = mixup_data(imgs, labels)
        logits = model(mixed_imgs)
        loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)
        
        # Log metrics every 10 steps
        if (step + 1) % 10 == 0:
            current_loss = np.mean(train_loss[-10:])
            current_acc = np.mean(train_accs[-10:])
            print(f"Epoch [{epoch+1}] Step [{step+1}/{len(train_loader)}] "
                  f"Loss: {current_loss:.4f} Acc: {current_acc:.4f}")
            
            # Log to wandb
            wandb.log({
                "train/step": epoch * len(train_loader) + step,
                "train/step_loss": current_loss,
                "train/step_acc": current_acc,
                "train/learning_rate": optimizer.param_groups[0]['lr']
            })
    
    scheduler.step()
    return np.mean(train_loss), np.mean(train_accs)

def valid_epoch(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = []
    valid_accs = []
    
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc.item())
            
    return np.mean(valid_loss), np.mean(valid_accs)

def calculate_class_weights(dataset_path):
    """
    sklearn의 compute_class_weight를 사용하여 클래스별 가중치를 계산하는 함수
    Args:
        dataset_path (str): 데이터셋 경로
    Returns:
        torch.Tensor: 각 클래스별 가중치
    """
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg'):
            label = int(filename.split('_')[0])
            labels.append(label)
    
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    
    return torch.FloatTensor(weights)

def main():
    # Initialize wandb
    wandb.init(
        project="food-classification",
        name="enhanced-classifier",
        config={
            "architecture": "EnhancedClassifier",
            "dataset": "Food-11",
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingWarmRestarts",
            "weight_decay": 0.01,
            "label_smoothing": 0.1
        }
    )
    
    # 하이퍼파라미터 설정
    batch_size = 32
    n_epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _dataset_dir = "/root/datasets/"
    
    # 클래스 가중치 계산
    class_weights = calculate_class_weights(os.path.join(_dataset_dir, "train")).to(device)
    
    # 데이터로더 설정
    train_set = FoodDataset(os.path.join(_dataset_dir, "train"), tfm=train_tfm)
    valid_set = FoodDataset(os.path.join(_dataset_dir, "validation"), tfm=test_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 모델 및 학습 설정
    model = EnhancedClassifier().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,  # 클래스 가중치 추가
        label_smoothing=0.1
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Watch model
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    # 학습 루프
    best_acc = 0
    patience = 20
    stale = 0
    _exp_name = "food_classification_enhanced"
    
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        valid_loss, valid_acc = valid_epoch(model, valid_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
        
        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": train_loss,
            "train/epoch_acc": train_acc,
            "valid/loss": valid_loss,
            "valid/acc": valid_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")
            print(f"Best model saved! ACC: {best_acc:.4f}")
            
            # Log best metrics to wandb
            wandb.run.summary["best_accuracy"] = best_acc
            wandb.run.summary["best_epoch"] = epoch + 1
            
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvement for {patience} epochs, stopping...")
                break
    
    # 테스트 데이터 예측
    test_set = FoodDataset(os.path.join(_dataset_dir, "test"), tfm=test_tfm, is_test=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model.eval()
    
    predictions = []
    file_ids = []
    
    with torch.no_grad():
        for data, _, file_id in tqdm(test_loader):
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            file_ids.extend(file_id)
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'ID': file_ids,
        'Category': predictions
    })
    submission.to_csv('submission.csv', index=False)
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()