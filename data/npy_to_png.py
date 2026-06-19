"""
npy_to_png.py

Script che converte immagini da formato .npy a .png senza perdita di qualità.
- Usa PNG (lossless compression)
- Mantiene la scala di grigi originale
- Preserva i valori minimi e massimo
"""

import os
import numpy as np
from PIL import Image
from glob import glob

def npy_to_png(npy_path, png_path=None):
    image = np.load(npy_path)
    
    print(f"Immagine caricata: {npy_path}")
    print(f"  Shape: {image.shape}")
    print(f"  Tipo: {image.dtype}")
    print(f"  Min: {image.min()}, Max: {image.max()}")
    
    if image.ndim > 2:
        if image.shape[0] in [1, 3, 4]:  
            image = image[0] if image.shape[0] == 1 else image
        else:
            image = image[..., 0]  
    
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = image * 255
        else:
            image = (image - image.min()) / (image.max() - image.min()) * 255
        
        image = image.astype(np.uint8)
    elif not np.issubdtype(image.dtype, np.uint8):
        image = image.astype(np.uint8)
    
    print(f"Dopo conversione:")
    print(f"  Tipo: {image.dtype}")
    print(f"  Min: {image.min()}, Max: {image.max()}")
    
    pil_image = Image.fromarray(image, mode='L')
    
    if png_path is None:
        png_path = npy_path.replace('.npy', '.png')
    
    pil_image.save(png_path, format='PNG', compress_level=0)  # compress_level=0 per massima qualità
    
    print(f"Immagine salvata: {png_path}")
    
    return png_path


def convert_folder(input_folder, output_folder=None):

    npy_files = glob(os.path.join(input_folder, "*.npy"))
    
    if not npy_files:
        print(f"Nessun file .npy trovato in {input_folder}")
        return
    
    print(f"Trovati {len(npy_files)} file .npy in {input_folder}")
    
    if output_folder is None:
        output_folder = input_folder
    
    os.makedirs(output_folder, exist_ok=True)
    
    for i, npy_path in enumerate(npy_files, 1):
        print(f"\n[{i}/{len(npy_files)}] Convertendo: {npy_path}")
        
        png_path = os.path.join(output_folder, os.path.basename(npy_path).replace('.npy', '.png'))
        
        try:
            npy_to_png(npy_path, png_path)
        except Exception as e:
            print(f"ERRORE: {e}")
    
    print(f"\n Conversione completata! File salvati in: {output_folder}")


if __name__ == "__main__":
    
    npy_to_png("/home/extter/sparse_mayo_light/data/pnp/results_pnp_180/pnp_001.npy")
    
    #convert_folder("../data/preprocessed/train", "../data/preprocessed/train_png")
    