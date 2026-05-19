import os
import numpy as np 
from tqdm import tqdm

import torch 
import torch.nn as nn 
from torch.optim import Adam


from torch.utils.data import Dataloader

from methods.pnp.dataset import GaussianNoiseDataset
from methods.end_to_end.unet import UNet

# PATH CONFIGURATION 
gt_dir = "../../data/mayo" #DA CAMBIARE 
json_path = "../../data/mayo/split.json"#DA CAMBIARE 


# PARAMETERS CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 50 
LR = 1e-4

# DATALOADER INITIALIZATION 
train_dataset = GaussianNoiseDataset(gt_dir, json_path, split="train")
train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = GaussianNoiseDataset(gt_dir, json_path, split="val")
val_dataloader = Dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = UNet(in_channels=1, out_channels=1).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss() # Anche questa forse da cambiare 

scaler = torch.cuda.amp.GradScaler()

# TRAINING LOOP
best_val_loss = float('inf')



for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for noisy_images, clean_images in progress_bar:
        noisy_images = noisy_images.to(DEVICE)
        clean_images = clean_images.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_train_loss = train_loss / len(train_dataloader)
    
    model.eval()
    val_loss = 0.0
    val_progress_bar = tqdm(val_dataloader, leave=False)

    with torch.no_grad():
        for noisy_images, clean_images in val_dataloader:
            noisy_images = noisy_images.to(DEVICE)
            clean_images = clean_images.to(DEVICE)

            with torch.cuda.amp.autocast():
                outputs = model(noisy_images)
                loss = criterion(outputs, clean_images)
            
            val_loss += loss.item()
            val_progress_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})
    
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "pnp_denoiser_best.pth")
        print(f"-> New best model saved!)")

print("Training completed!")














