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
    """
    Converti un'immagine .npy a .png senza perdita di qualità.
    
    Args:
        npy_path: percorso del file .npy
        png_path: percorso(output) del file .png (opzionale, se None viene generato automaticamente)
    
    Returns:
        png_path: percorso del file .png salvato
    """
    # Carica il file npy
    image = np.load(npy_path)
    
    print(f"Immagine caricata: {npy_path}")
    print(f"  Shape: {image.shape}")
    print(f"  Tipo: {image.dtype}")
    print(f"  Min: {image.min()}, Max: {image.max()}")
    
    # Se l'immagine ha più dimensioni (es. [C, H, W]), prendi solo il primo canale
    if image.ndim > 2:
        if image.shape[0] in [1, 3, 4]:  # Se è un canale o RGB/RGBA
            image = image[0] if image.shape[0] == 1 else image
        else:
            image = image[..., 0]  # Prendi il primo canale se è molto grande
    
    # Se l'immagine è float, normalizza a [0, 255] e converti a uint8
    if np.issubdtype(image.dtype, np.floating):
        # Se i valori sono già in [0, 1], moltiplica per 255
        if image.max() <= 1.0:
            image = image * 255
        else:
            # Normalizza rispetto al range minimo/massimo
            image = (image - image.min()) / (image.max() - image.min()) * 255
        
        image = image.astype(np.uint8)
    elif not np.issubdtype(image.dtype, np.uint8):
        # Converti a uint8 da altri tipi integer
        image = image.astype(np.uint8)
    
    print(f"Dopo conversione:")
    print(f"  Tipo: {image.dtype}")
    print(f"  Min: {image.min()}, Max: {image.max()}")
    
    # Creazione dell'immagine PIL (grayscale)
    pil_image = Image.fromarray(image, mode='L')
    
    # Se png_path non è specificato, genera automaticamente
    if png_path is None:
        png_path = npy_path.replace('.npy', '.png')
    
    # Salva come PNG (lossless) con compressione massima
    pil_image.save(png_path, format='PNG', compress_level=0)  # compress_level=0 per massima qualità
    
    print(f"Immagine salvata: {png_path}")
    
    return png_path


def convert_folder(input_folder, output_folder=None):
    """
    Converti tutte le immagini .npy in una folder a .png.
    
    Args:
        input_folder: percorso della folder input con file .npy
        output_folder: percorso della folder output (opzionale)
    """
    # Trova tutti i file .npy
    npy_files = glob(os.path.join(input_folder, "*.npy"))
    
    if not npy_files:
        print(f"Nessun file .npy trovato in {input_folder}")
        return
    
    print(f"Trovati {len(npy_files)} file .npy in {input_folder}")
    
    # Se output_folder non è specificato, usa la stessa folder input
    if output_folder is None:
        output_folder = input_folder
    
    # Crea la folder output se non esiste
    os.makedirs(output_folder, exist_ok=True)
    
    # Converti tutti i file
    for i, npy_path in enumerate(npy_files, 1):
        print(f"\n[{i}/{len(npy_files)}] Convertendo: {npy_path}")
        
        # Genera il percorso output
        png_path = os.path.join(output_folder, os.path.basename(npy_path).replace('.npy', '.png'))
        
        try:
            npy_to_png(npy_path, png_path)
        except Exception as e:
            print(f"ERRORE: {e}")
    
    print(f"\n✓ Conversione completata! File salvati in: {output_folder}")


if __name__ == "__main__":
    # ================= DUE OPZIONI =================
    
    # OPZIONE 1: Converti un singolo file
    npy_to_png("/home/catas/sparse_mayo_light/data/preprocessed/test/test_C081_0.npy")
    
    # OPZIONE 2: Converti tutta una folder
    #convert_folder("../data/preprocessed/train", "../data/preprocessed/train_png")
    
    # =================