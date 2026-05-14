"""
Training loop for CT reconstruction using UNet

Usage: 
    python train.py --fbp_dir data/fbp/angle_090 --tv_dir data/tv/angle_090 #poi cata sono da cambiare ovviamente
"""

import argparse
import os 
import time 
import json 

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch 
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from dataset import get_dataloaders

from evaluation.metrics import SSIM, PSNR
from third_party.ippy.operators import *
from utilities import *
from dataset import *
from losses import MixedLoss






def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    projector: object = None,
    scheduler: object = None,
    n_epochs: int = 50,
    save_each: int | None = None,
    weights_path: str | None = None,
    device: str = "cpu",
) -> dict:
    r"""
    Train a given pytorch model on an input training set, tracking validation metrics.
    Saves the best model based on validation loss.
    """
    print(f"Training NN model for {n_epochs} epochs on {device}.")
    
    loss_history = {"train": [], "val": []}
    ssim_history = {"train": [], "val": []}
    
    best_val_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for epoch in range(n_epochs):
        
        model.train()
        epoch_train_loss = 0.0
        epoch_train_ssim = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        start_time = time.time()
        
        for t, (x_sino_noisy, x_tv) in enumerate(progress_bar, start=1):
            x_sino_noisy, x_tv = x_sino_noisy.to(device), x_tv.to(device)
            x_fbp = projector.FBP(x_sino_noisy)

            optimizer.zero_grad()
            y_pred = model(x_fbp)
            
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                y_pred = model(x_fbp)
                loss = loss_fn(y_pred, x_tv)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
            ssim_val = SSIM(y_pred.detach(), x_tv.detach())
            epoch_train_ssim += ssim_val.item() if hasattr(ssim_val, "item") else ssim_val

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_ssim = epoch_train_ssim / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_ssim = 0.0
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        with torch.no_grad():
            for v, (x_sino_noisy_val, x_tv_val) in enumerate(progress_bar_val, start=1):
                x_sino_noisy_val, x_tv_val = x_sino_noisy_val.to(device), x_tv_val.to(device)

                x_fbp_val = projector.FBP(x_sino_noisy_val)
                
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    y_val_pred = model(x_fbp_val)
                    val_loss = loss_fn(y_val_pred, x_tv_val)
                    
                epoch_val_loss += val_loss.item()
                ssim_val = SSIM(y_val_pred.detach(), x_tv_val.detach())
                epoch_val_ssim += ssim_val.item() if hasattr(ssim_val, "item") else ssim_val
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_ssim = epoch_val_ssim / len(val_loader)

        # Update history
        loss_history["train"].append(avg_train_loss)
        loss_history["val"].append(avg_val_loss)
        ssim_history["train"].append(avg_train_ssim)
        ssim_history["val"].append(avg_val_ssim)
        
        if scheduler is not None:
            scheduler.step()

        # Verbose
        time_str = formatted_time(start_time)
        print(
            f"({time_str}) Epoch {epoch+1:03d}/{n_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} SSIM: {avg_train_ssim:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} SSIM: {avg_val_ssim:.4f}"
        )

        # Save Best Model Strategy (Migliore di save_each per evitare overfitting)
        if weights_path is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Utilizza la tua funzione di salvataggio
            save(model, weights_path)
            print(f" -> Checkpoint saved! New best val loss: {best_val_loss:.4f}")
            
        # Optional period save 
        elif save_each is not None and (epoch + 1) % save_each == 0:
             # Modifica il nome per non sovrascrivere il best model
             save(model, f"{weights_path}_epoch_{epoch+1}")

    print("\nTraining completed!")
    return {"loss": loss_history, "ssim": ssim_history}


def save(model: nn.Module, weights_path: str):
    # La tua funzione rimane identica
    create_path_if_not_exists(weights_path)
    model_config = get_config(model)
    with open(f"{weights_path}/config.json", "w") as fp:
        json.dump(model_config, fp, indent=2)
    torch.save(model.state_dict(), f"{weights_path}/weights.pth")


def load(weights_path: str):
    # La tua funzione rimane identica
    with open(f"{weights_path}/config.json") as fp:
        model_config = json.load(fp)
    model = UNet(**model_config)
    model.load_state_dict(torch.load(f"{weights_path}/weights.pth", map_location="cpu"))
    return model


if __name__ == "__main__":
    # --- PARSING DEGLI ARGOMENTI DA TERMINALE ---
    parser = argparse.ArgumentParser(description="Train UNet for CT Reconstruction")
    
    # Parametro principale per automatizzare i dataset
    parser.add_argument("--angle", type=str, required=True, 
                        help="Angle configuration (e.g., 180, 090, 060, 045)")
    
    # Percorsi base (puoi personalizzarli o sovrascriverli)
    parser.add_argument("--base_data_dir", type=str, default="data", help="Base directory for datasets")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Where to save models")
    
    # Iperparametri
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()

    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Costruzione dei path automatici in base all'angolo
    weights_path = os.path.join(args.save_dir, f"unet_angle_{args.angle}")

    print(f"Starting training configuration for angle {args.angle}")
    print(f"Model checkpoints will be saved in: {weights_path}")

    # 3. Creazione Dataloaders (Usa la funzione che hai creato per i pazienti)
    train_loader, val_loader, _ = get_dataloaders(
        base_data_dir=args.base_data_dir, 
        angle=args.angle, 
        batch_size=args.batch_size
    )

    n_angles = int(args.angle)
    angles_array = np.linspace(0, np.pi, n_angles, endpoint=False)
    projector = CTProjector(img_shape=(256, 256), det_size=256, angles=angles_array, force_cpu=False)
    
    # 4. Inizializzazione Modello e Ottimizzatore
    model = UNet(ch_in=1, ch_out=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 5. Avvio del Training
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,     
        optimizer=optimizer,
        loss_fn=nn.MSELoss(), 
        projector=projector,
        scheduler=scheduler,
        save_each=5,
        n_epochs=args.epochs,
        weights_path=weights_path,
        device=device
    )
    
    # Salva la history in un file JSON per i tuoi grafici finali
    with open(f"{weights_path}/history.json", "w") as f:
        json.dump(history, f)