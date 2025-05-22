import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class TrainDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted([os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.endswith(".jpg")])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

def calculate_mean_std():
    data_dir = "./data/train"  # Update this path if needed
    batch_size = 32
    
    # Create dataset and dataloader
    dataset = TrainDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    
    # Initialize variables for mean and std calculation
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_batches = len(dataloader)
    
    # Calculate mean
    print("Calculating mean and std...")
    for data in tqdm(dataloader):
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5
    
    print(f"\nDataset mean: {mean}")
    print(f"Dataset std: {std}")
    
    return mean, std

if __name__ == "__main__":
    mean, std = calculate_mean_std()
    breakpoint()