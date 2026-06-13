import os
import numpy as np 
from tqdm import tqdm

import torch 
import torch.nn as nn 
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader

from methods.pnp.dataset import GaussianNoiseDataset
from methods.end_to_end_2.unet import SimpleUNet

# PATH CONFIGURATION 
gt_train = "../../data/mayo/train" #DA CAMBIARE 
gt_val = "../../data/mayo/validation" #DA CAMBIARE



# PARAMETERS CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 50 
LR = 1e-3

# DATALOADER INITIALIZATION 
train_dataset = GaussianNoiseDataset(gt_train, split="train")
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = GaussianNoiseDataset(gt_val, split="val")
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SimpleUNet(ch_in=1, ch_out=1, base_ch=16).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, patience=2)
criterion_l1 = nn.L1Loss()
criterion_l2 = nn.MSELoss() 

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
            
            loss_l1 = criterion_l1(outputs, clean_images)
            loss_l2 = criterion_l2(outputs, clean_images)
            loss = 0.5 * loss_l1 + 0.5 * loss_l2
        
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                v_loss_l1 = criterion_l1(outputs, clean_images)
                v_loss_l2 = criterion_l2(outputs, clean_images)
                loss = 0.5 * v_loss_l1 + 0.5 * v_loss_l2
            
            val_loss += loss.item()
            val_progress_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})
    
    avg_val_loss = val_loss / len(val_dataloader)

    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {current_lr}")

    print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "pnp_denoiser_best.pth")
        print(f"-> New best model saved!)")

print("Training completed!")














