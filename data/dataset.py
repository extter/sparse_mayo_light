#ERRORE: QUESTO FILE È STATO MODIFICATO RECENTEMENTE. CAPIRE IL COMMIT DELL'ERRORE

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


def find_raw_file(raw_root, filename):
    matches = list(Path(raw_root).rglob(filename))
    if len(matches) == 0:
        raise FileNotFoundError(f"Ground truth non trovata per {filename} in {raw_root}")
    if len(matches) > 1:
        raise RuntimeError(f"Trovati più file per {filename}: {matches}")
    return matches[0]


def main():
    base_data_dir = "data/dataset_nn"
    angle = "180"
    split = "test"

    weights_path = f"checkpoints/unet_angle_{angle}"
    reco_dir = Path("data/reco") / f"angles_{int(angle)}" / split
    raw_dir = Path("data/raw") / split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders(
        base_data_dir=base_data_dir,
        angle=angle,
        batch_size=1
    )

    x_sino, x_tv = next(iter(test_loader))

    sample_idx = 0
    sample_path = Path(test_loader.dataset.input_files[sample_idx])
    filename = sample_path.name
    print(f"Sample usato: {filename}")

    x_sino = x_sino.to(device, non_blocking=True)

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

    raw_path = find_raw_file(raw_dir, filename)

    gt_img = np.load(raw_path).astype(np.float32)
    reco_img = np.load(reco_path).astype(np.float32)
    pred_img = pred.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gt_img, cmap="gray")
    axes[0].set_title("Ground truth")
    axes[0].axis("off")

    axes[1].imshow(reco_img, cmap="gray")
    axes[1].set_title("Reco")
    axes[1].axis("off")

    axes[2].imshow(pred_img, cmap="gray")
    axes[2].set_title("Inference")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()