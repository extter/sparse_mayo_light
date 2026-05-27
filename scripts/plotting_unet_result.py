import os
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.end_to_end.unet import UNet
from third_party.ippy.operators import CTProjector
from methods.end_to_end.dataset import get_dataloaders


def load_model(weights_path, device):
    with open(os.path.join(weights_path, "config.json"), "r") as f:
        config = json.load(f)

    model = UNet(**config)
    model.load_state_dict(torch.load(os.path.join(weights_path, "weights.pth"), map_location=device))
    model.to(device)
    model.eval()
    return model


def build_raw_path(raw_root, filename):
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Filename non nel formato atteso: {filename}")

    patient_id = parts[1]
    slice_idx = parts[2]

    raw_path = Path(raw_root) / patient_id / f"{slice_idx}.png"

    if not raw_path.exists():
        raise FileNotFoundError(f"Ground truth non trovata: {raw_path}")

    return raw_path


def main():
    base_data_dir = "data/dataset_nn"
    angle = "060"
    split = "test"

    weights_path = f"checkpoints/unet_angle_{angle}/20260527_115105/"
    reco_dir = Path("data/reco") / f"angles_{int(angle)}" / split
    raw_dir = Path("data/raw") / split
    preprocessed_dir = Path("data/preprocessed") / split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders(
        base_data_dir=base_data_dir,
        angle=angle,
        batch_size=1
    )

    x_sino, x_tv = next(iter(test_loader))

    sample_idx = 100
    sample_path = Path(test_loader.dataset.input_files[sample_idx])
    filename = sample_path.name
    print(f"Sample usato: {filename}")

    # Carica direttamente il sample corretto
    x_sino, x_tv = test_loader.dataset[sample_idx]
    x_sino = x_sino.unsqueeze(0).to(device, non_blocking=True)  # aggiungi batch dimension

    model = load_model(weights_path, device)

    n_angles = int(angle)
    angles_array = np.linspace(0, np.pi, n_angles, endpoint=False)
    projector = CTProjector(
        img_shape=(256, 256),
        det_size=512,
        angles=angles_array,
        force_cpu=(device.type == "cpu")
    )

    with torch.no_grad():
        x_fbp = projector.FBP(x_sino)
        pred = model(x_fbp)
        pred = torch.clamp(pred, 0.0, 1.0)

    reco_path = reco_dir / filename
    if not reco_path.exists():
        raise FileNotFoundError(f"Reco non trovata: {reco_path}")

    preprocessed_path = preprocessed_dir / filename
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed non trovata: {preprocessed_path}")

    raw_path = build_raw_path(raw_dir, filename)

    gt_img = plt.imread(raw_path).astype(np.float32)
    if gt_img.ndim == 3:
        gt_img = gt_img[..., 0]

    preprocessed_img = np.load(preprocessed_path).astype(np.float32)
    reco_img = np.load(reco_path).astype(np.float32)
    pred_img = pred.squeeze().detach().cpu().numpy()

    with torch.no_grad():
        x_fbp = projector.FBP(x_sino)
        # Normalizzazione per immagine
        x_min = x_fbp.amin(dim=(1,2,3), keepdim=True)
        x_max = x_fbp.amax(dim=(1,2,3), keepdim=True)
        x_fbp_norm = (x_fbp - x_min) / (x_max - x_min + 1e-8)
        pred = model(x_fbp_norm)
        pred = torch.clamp(pred, 0.0, 1.0)

    fbp_img = x_fbp_norm.squeeze().detach().cpu().numpy()


    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(gt_img, cmap="gray")
    axes[0].set_title("Ground truth")
    axes[0].axis("off")
    axes[1].imshow(preprocessed_img, cmap="gray")
    axes[1].set_title("Preprocessed")
    axes[1].axis("off")
    axes[2].imshow(reco_img, cmap="gray")
    axes[2].set_title("Reco")
    axes[2].axis("off")
    axes[3].imshow(fbp_img, cmap="gray")
    axes[3].set_title("FBP corrotta")
    axes[3].axis("off")
    axes[4].imshow(pred_img, cmap="gray")
    axes[4].set_title("Inference")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()