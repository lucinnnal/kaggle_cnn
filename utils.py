# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from sklearn.utils.class_weight import compute_class_weight

class FoodDataset(Dataset):
    def __init__(self, path, tfm=None, files=None, is_test=False):
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

def visualize_label_distribution(dataset_path):
    """
    데이터셋의 라벨 분포를 시각화하는 함수
    Args:
        dataset_path (str): 데이터셋 경로
    """
    # 라벨 카운트를 저장할 딕셔너리
    label_counts = {i: 0 for i in range(11)}  # 0-10까지의 클래스
    
    # 모든 이미지 파일에 대해 라벨 카운트
    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg'):
            label = int(filename.split('_')[0])
            label_counts[label] += 1
    
    # 시각화
    plt.figure(figsize=(12, 6))
    bars = plt.bar(label_counts.keys(), label_counts.values())
    
    # 각 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.title('Distribution of Food Categories')
    plt.xlabel('Category')
    plt.ylabel('Number of Images')
    plt.grid(True, alpha=0.3)
    
    # 전체 이미지 수
    total_images = sum(label_counts.values())
    plt.text(0.02, 0.95, f'Total Images: {total_images}', 
             transform=plt.gca().transAxes)
    
    return label_counts

def calculate_class_weights(dataset_path):
    """
    sklearn의 compute_class_weight를 사용하여 클래스별 가중치를 계산하는 함수
    Args:
        dataset_path (str): 데이터셋 경로
    Returns:
        torch.Tensor: 각 클래스별 가중치
    """
    # 모든 라벨 수집
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg'):
            label = int(filename.split('_')[0])
            labels.append(label)
    
    # 고유한 클래스 레이블
    classes = np.unique(labels)
    
    # compute_class_weight를 사용하여 가중치 계산
    weights = compute_class_weight(
        class_weight='balanced',  # 'balanced' 옵션으로 자동 균형 조정
        classes=classes,
        y=labels
    )
    
    # numpy array를 PyTorch tensor로 변환
    class_weights = torch.FloatTensor(weights)
    
    return class_weights

# 사용 예시
if __name__ == "__main__":
    train_path = "./data/train"  # 경로는 실제 데이터셋 위치로 수정하세요
    label_counts = visualize_label_distribution(train_path)
    plt.show()
    
    # 각 클래스별 비율 출력
    total = sum(label_counts.values())
    print("\nClass distribution:")
    for label, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"Class {label}: {count} images ({percentage:.2f}%)")
    
    # 클래스 가중치 계산
    class_weights = calculate_class_weights(train_path)
    
    print("\nClass weights:")
    for i, weight in enumerate(class_weights):
        print(f"Class {i}: {weight:.4f}")
    
    # Cross Entropy Loss에 가중치 적용 예시
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.1
    )