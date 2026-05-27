import os 
import json
import numpy as np 
import torch 
from torch.utils.data import Dataset

class GaussianNoiseDataset(Dataset):
    def __init__ (self, gt_dir, split="train"):
        self.gtdir = gt_dir
        self.split = split
        
        self.filenames = sorted([f for f in os.listdir(self.gtdir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        file_path = os.path.join(self.gtdir, filename)

        clean_image = np.load(file_path)
        clean_tensor = torch.from_numpy(clean_image).float().unsqueeze(0) # Shape: (1,256,256)

        if self.split == 'train':
            sigma = np.random.uniform(0.01, 0.05)
            noise = torch.randn_like(clean_tensor) * sigma
            noisy_tensor = clean_tensor + noise
        else:
            noise = torch.randn_like(clean_tensor) * 0.03
            noisy_tensor = clean_tensor + noise

        return noisy_tensor, clean_tensor
    
class TCDataset(Dataset):
    def __init__(self, sino_dir, gt_dir):
        """
        Args:
            sino_dir (str): Percorso alla cartella contenente i sinogrammi rumorosi.
            gt_dir (str): Percorso alla cartella contenente le immagini pulite (Ground Truth).
        """
        self.sino_dir = sino_dir
        self.gt_dir = gt_dir
        
        # Legge i nomi dei file e li ordina per essere sicuro che combacino perfettamente
        self.sino_files = sorted([f for f in os.listdir(sino_dir) if f.endswith('.npy')])
        self.gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])
        
        # Controllo di sicurezza per evitare disallineamenti tra input e target
        if len(self.sino_files) != len(self.gt_files):
            print(f"⚠️ Attenzione: Trovati {len(self.sino_files)} sinogrammi e {len(self.gt_files)} immagini GT!")

    def __len__(self):
        # Dice a PyTorch quante slice ci sono in totale nel dataset
        return len(self.sino_files)

    def __getitem__(self, idx):
        # 1. Recupera i nomi dei file corrispondenti all'indice richiesto
        sino_name = self.sino_files[idx]
        gt_name = self.gt_files[idx]
        
        # 2. Costruisce i percorsi completi sul disco
        sino_path = os.path.join(self.sino_dir, sino_name)
        gt_path = os.path.join(self.gt_dir, gt_name)
        
        # 3. Carica i file binari da disco alla memoria RAM al volo (Lazy Loading)
        sino_array = np.load(sino_path)
        gt_array = np.load(gt_path)
        
        # 4. Converte in Tensori PyTorch (float32 è lo standard per le reti neurali)
        sino_tensor = torch.tensor(sino_array, dtype=torch.float32)
        gt_tensor = torch.tensor(gt_array, dtype=torch.float32)
        
        # 5. AGGIUNGE IL CANALE [1, H, W] richiesto da PyTorch per le convoluzioni
        # Se i tuoi array sono già 3D (es. shape [1, 256, 256]), salta questo step.
        if sino_tensor.ndim == 2:
            sino_tensor = sino_tensor.unsqueeze(0)
        if gt_tensor.ndim == 2:
            gt_tensor = gt_tensor.unsqueeze(0)
            
        return sino_tensor, gt_tensor
    
    
        
            


    