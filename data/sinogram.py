"""
sinogram.py

Utilities per generare sinogrammi clean e corrupted
a partire dalle immagini preprocessate del task Sparse-view CT.
"""

import os
import numpy as np
import torch

from third_party.ippy import operators, _utilities


def get_device():
    """Restituisce il device disponibile."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_operators_dict(angle_configs, image_size, det_size):
    """
    Crea un dizionario di operatori CTProjector, uno per ogni configurazione angolare.

    Args:
        angle_configs (list[int]): lista dei numeri di angoli, es. [180, 90, 60, 45]
        image_size (int): dimensione immagine quadrata
        det_size (int): dimensione detector

    Returns:
        dict[int, CTProjector]
    """
    operators_dict = {}

    for n_angles in angle_configs:
        K = operators.CTProjector(
            img_shape=(image_size, image_size),
            angles=np.linspace(0, np.pi, n_angles, endpoint=False),
            det_size=det_size,
            geometry="parallel",
            force_cpu=False,
        )
        operators_dict[n_angles] = K

    return operators_dict


def generate_sinogram_pair(img_np, projector, noise_level, device):
    """
    Genera sinogramma clean e corrupted da una singola immagine preprocessata.

    Args:
        img_np (np.ndarray): immagine shape (H, W), float32, valori in [0,1]
        projector: operatore CTProjector
        noise_level (float): livello di rumore gaussiano
        device (torch.device): cpu o cuda

    Returns:
        tuple[np.ndarray, np.ndarray]: (clean_sinogram, corrupted_sinogram)
    """
    img_tensor = (
        torch.from_numpy(img_np)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    clean_sinogram = projector(img_tensor)
    corrupted_sinogram = clean_sinogram + _utilities.gaussian_noise(
        clean_sinogram,
        noise_level=noise_level
    )

    clean_np = clean_sinogram[0, 0].detach().cpu().numpy().astype(np.float32)
    corrupted_np = corrupted_sinogram[0, 0].detach().cpu().numpy().astype(np.float32)

    return clean_np, corrupted_np


def build_output_filename(img_path, preprocessed_split_path):
    rel_path = os.path.relpath(img_path, preprocessed_split_path)
    filename = os.path.splitext(rel_path.replace(os.sep, "_"))[0] + ".npy"
    return filename


def save_sinograms_for_split(
    image_paths,
    split_name,
    preprocessed_path,
    sinogram_clean_path,
    sinogram_corrupted_path,
    operators_dict,
    angle_configs,
    noise_level,
    device,
):

    split_preprocessed_path = os.path.join(preprocessed_path, split_name)

    for img_path in image_paths:
        img_np = np.load(img_path).astype(np.float32)
        filename = build_output_filename(img_path, split_preprocessed_path)

        for n_angles in angle_configs:
            projector = operators_dict[n_angles]

            clean_np, corrupted_np = generate_sinogram_pair(
                img_np=img_np,
                projector=projector,
                noise_level=noise_level,
                device=device,
            )

            clean_path = os.path.join(
                sinogram_clean_path,
                split_name,
                f"angles_{n_angles}",
                filename,
            )
            corrupted_path = os.path.join(
                sinogram_corrupted_path,
                split_name,
                f"angles_{n_angles}",
                filename,
            )

            np.save(clean_path, clean_np)
            np.save(corrupted_path, corrupted_np)

    print(f"Saved sinograms for {split_name}: {len(image_paths)} images")