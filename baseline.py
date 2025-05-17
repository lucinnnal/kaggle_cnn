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

# This is for the progress bar.
from tqdm.auto import tqdm
import random

# Import wandb at the top of the file
import wandb

myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image into a fixed shape (height = width = 512)
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


class FoodDataset(Dataset):

    def __init__(self, path, tfm=test_tfm, files=None, is_test=False):
        super(FoodDataset).__init__()
        self.path = path
        self.is_test = is_test
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        file_id = os.path.basename(fname).split(".")[0]  # Extract the ID from the filename
        if self.is_test:
            return im, -1, file_id  # Return file_id only for test set
        else:
            label = int(fname.split("/")[-1].split("_")[0])
            return im, label


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # input dimension [3, 256, 256]
        self.cnn = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, 3, 1, 1),      # [64, 256, 256]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [64, 128, 128]
            
            # Second block
            nn.Conv2d(64, 128, 3, 1, 1),    # [128, 128, 128]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [128, 64, 64]
            
            # Third block
            nn.Conv2d(128, 256, 3, 1, 1),   # [256, 64, 64]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),   # [256, 64, 64]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [256, 32, 32]
            
            # Fourth block
            nn.Conv2d(256, 512, 3, 1, 1),   # [512, 32, 32]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),   # [512, 32, 32]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [512, 16, 16]
            
            # Fifth block
            nn.Conv2d(512, 512, 3, 1, 1),   # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),   # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [512, 8, 8]

            # Sixth block
            nn.Conv2d(512, 512, 3, 1, 1),   # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [512, 4, 4]
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # 과적합 방지
            nn.Linear(512 * 4 * 4, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),  # 과적합 방지
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

batch_size = 64
_dataset_dir = "./data"
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = FoodDataset(os.path.join(_dataset_dir,"train"), tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# The number of training epochs and patience.
n_epochs = 3
patience = 300  # If no improvement in 'patience' epochs, early stop

# Initialize a model, and put it on the device specified.
model = Classifier().to(device)

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Initialize wandb before training
wandb.init(
    project="food-classification",
    config={
        "architecture": "Custom_CNN",
        "dataset": "Food-11",
        "epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "optimizer": "AdamW"
    }
)

# Watch the model
wandb.watch(model)

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0
_exp_name = "food_classification_experiment_Simple"

for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # imgs = imgs.half()
        # print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Log training metrics
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
    })

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        # break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Log validation metrics
    wandb.log({
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
    })

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # save models
    if valid_acc > best_acc:
        # Log best model metrics
        wandb.log({
            "best_valid_acc": valid_acc,
            "best_epoch": epoch + 1
        })
        
        # Save model to wandb
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"{_exp_name}_best.ckpt"))

        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")  # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

# Close wandb run at the end
wandb.finish()

test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm, is_test=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()
prediction = []
file_ids = []  # To store file IDs for each image

with torch.no_grad():
    for data,_,file_id in test_loader:
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()
        file_ids += file_id  # Append original IDs


#create test csv
df = pd.DataFrame()
df["ID"] = file_ids
df["Category"] = prediction
df.to_csv("submission.csv",index = False)