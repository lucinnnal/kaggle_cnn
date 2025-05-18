import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn as nn
import os
from baseline import ImprovedClassifier, FoodDataset, train_tfm, test_tfm

def resume_training():
    # Hyperparameters
    batch_size = 128  # Match original training batch size
    n_epochs = 31  # Additional epochs
    patience = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _dataset_dir = "/content/data/"
    _exp_name = "food_classification_improved"

    # Construct datasets
    train_set = FoodDataset(os.path.join(_dataset_dir, "train"), tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_set = FoodDataset(os.path.join(_dataset_dir, "validation"), tfm=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model and components
    model = ImprovedClassifier(num_classes=11).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    scheduler = torch.optim.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-4)

    # Load checkpoint
    checkpoint = torch.load(f"{_exp_name}_best.ckpt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']

    print(f"Resuming from epoch {start_epoch} with best accuracy: {best_acc:.4f}")

    # Initialize wandb
    wandb.init(
        project="food-classification",
        name=f"{_exp_name}_resumed",
        config={
            "architecture": "ImprovedClassifier",
            "resumed_from_epoch": start_epoch,
            "total_epochs": start_epoch + n_epochs,
            "batch_size": batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR"
        }
    )

    # Watch model
    wandb.watch(model, log="all")

    # Training tracking variables
    stale = 0

    # Training and validation loop
    for epoch in range(n_epochs):
        current_epoch = start_epoch + epoch + 1
        
        # ---------- Training ----------
        model.train()
        train_loss = []
        train_accs = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {current_epoch}/{start_epoch+n_epochs} [Train]")
        
        for batch in train_pbar:
            imgs, labels = batch
            
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc.item())
            
            train_pbar.set_postfix(
                {"loss": sum(train_loss)/len(train_loss), "acc": sum(train_accs)/len(train_accs)}
            )
        
        train_loss = sum(train_loss)/len(train_loss)
        train_acc = sum(train_accs)/len(train_accs)
        
        # ---------- Validation ----------
        model.eval()
        valid_loss = []
        valid_accs = []
        
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {current_epoch}/{start_epoch+n_epochs} [Valid]")
        
        with torch.no_grad():
            for batch in valid_pbar:
                imgs, labels = batch
                logits = model(imgs.to(device))
                loss = criterion(logits, labels.to(device))
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
                
                valid_loss.append(loss.item())
                valid_accs.append(acc.item())
                
                valid_pbar.set_postfix(
                    {"loss": sum(valid_loss)/len(valid_loss), "acc": sum(valid_accs)/len(valid_accs)}
                )
        
        valid_loss = sum(valid_loss)/len(valid_loss)
        valid_acc = sum(valid_accs)/len(valid_accs)
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": current_epoch
        })
        
        # Print epoch info
        print(f"[ Train | {current_epoch:03d}/{start_epoch+n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        print(f"[ Valid | {current_epoch:03d}/{start_epoch+n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
        # Update scheduler
        scheduler.step()
        
        # Check for improvement
        if valid_acc > best_acc:
            print(f"Best model found at epoch {current_epoch}, saving model")
            best_acc = valid_acc
            stale = 0
            
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, f"{_exp_name}_resumed_best.ckpt")
            
            wandb.run.summary["best_accuracy"] = best_acc
            wandb.run.summary["best_epoch"] = current_epoch
        else:
            stale += 1
            print(f"No improvement in validation accuracy for {stale} epochs")
            if stale >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    wandb.finish()

if __name__ == "__main__":
    resume_training()