"""
Costruction of dataloader for end-to-end architecture for CT reconstruction of the Mayo dataset
using TV images as targets.

Expected directory structure:
    data/nn_dataset/
                     train/
                        sinograms/ 
                            angle_090/
                                img_001.npy
                                ...
                        tv/
                            angle_090/
                     val/
                        sinograms/
                        tv/
                     test/
                        sinograms/
                        tv/
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class CTDataset(Dataset): 
    def __init__(self, input_dir: str, target_dir: str):
        """
        Carica le coppie (input, target) leggendo tutti i file .npy nelle cartelle specificate.
        """
        input_paths = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
        self.input_files = []
        self.target_files = []
        
        # Appaiamento sicuro dei file tramite basename
        for f_path in input_paths:
            basename = os.path.basename(f_path)
            target_path = os.path.join(target_dir, basename)
            if os.path.exists(target_path):
                self.input_files.append(f_path)
                self.target_files.append(target_path)
            else:
                print(f"Warning: File target mancante per {basename}. Verrà ignorato.")

        assert len(self.input_files) > 0, f"Nessun file .npy trovato in {input_dir}!"

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Caricamento file binari (preserva i float32 essenziali per la TV)
        x = np.load(self.input_files[idx]).astype(np.float32)
        y = np.load(self.target_files[idx]).astype(np.float32)

        # Converti in tensori e aggiungi la dimensione del canale (1, 256, 256)
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)

        return x, y


def get_dataloaders(base_data_dir: str, angle: str, batch_size: int = 8, num_workers: int = 4):
    """
    Costruisce e restituisce direttamente i tre Dataloader.
    base_data_dir: la cartella radice (es. 'data')
    angle: l'angolo sotto forma di stringa (es. '090')
    """
    
    # 1. Costruisci i percorsi esatti per TRAIN
    train_sino = os.path.join(base_data_dir, "train", "sinograms", f"angle_{angle}")
    train_tv   = os.path.join(base_data_dir, "train", "tv", f"angle_{angle}")
    
    # 2. Costruisci i percorsi esatti per VAL
    val_sino = os.path.join(base_data_dir, "val", "sinograms", f"angle_{angle}")
    val_tv   = os.path.join(base_data_dir, "val", "tv", f"angle_{angle}")
    
    # 3. Costruisci i percorsi esatti per TEST
    test_sino = os.path.join(base_data_dir, "test", "sinograms", f"angle_{angle}")
    test_tv   = os.path.join(base_data_dir, "test", "tv", f"angle_{angle}")

    # 4. Inizializza i Dataset
    train_ds = CTDataset(train_sino, train_tv)
    val_ds   = CTDataset(val_sino, val_tv)
    test_ds  = CTDataset(test_sino, test_tv)

    print(f"Dataset caricato (Angolo {angle}) -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # 5. Crea i DataLoader
    # pin_memory=True velocizza il passaggio dei dati CPU -> GPU
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    # Il test loader di solito ha batch_size=1 per fare calcoli più precisi in fase di inferenza
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_workers) 

    return train_loader, val_loader, test_loader
