"""
Runna il training end-to-end per tutti e 4 i numeri di angoli,
poi fa l'inference e visualizza i risultati su tutto il test set.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

sys.path.append(str(Path(__file__).resolve().parents[1]))
from methods.end_to_end_2 import (
    SimpleUNet,
    get_dataloaders,
    get_ct_operator,
    get_device,
    plot_loss_curves,
    plot_results,
    train,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
book_root = Path(__file__).resolve().parents[1]
data_root = (book_root / 'data').resolve()
weights_dir = book_root / 'methods' / 'end_to_end_2' / 'weights'
weights_dir.mkdir(exist_ok=True)

device = get_device()

ANGLE_CONFIGS = [45, 60, 90, 180]
NUM_EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-3
DATA_SHAPE = 256

# Scegli qui la loss: 'mse' | 'ssim' | 'ssim_custom' | 'mixed'
LOSS_NAME = 'mixed'

print('Device:', device)
print('Data root:', data_root)
print('Weights dir:', weights_dir)
print('Loss:', LOSS_NAME)


def visualize_sample(model, dataset, K, device, n_angles, title=''):
    """Visualizza una tripla (raw, reco gt, FBP, UNet) dal primo sample del dataset."""
    model.eval()
    x_true, y_delta, target = dataset[0]
    x_true  = x_true.unsqueeze(0).to(device)
    y_delta = y_delta.unsqueeze(0).to(device)
    target  = target.unsqueeze(0).to(device)

    with torch.no_grad():
        x_fbp = torch.clamp(K.FBP(y_delta), min=0.0, max=1.0)
        x_rec = model(x_fbp)

    mse_fbp  = torch.mean((x_fbp - target) ** 2).item()
    mse_unet = torch.mean((x_rec - target) ** 2).item()
    ssim_fbp  = structural_similarity_index_measure(x_fbp, target, data_range=1.0).item()
    ssim_unet = structural_similarity_index_measure(x_rec, target, data_range=1.0).item()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(x_true.cpu().squeeze(), cmap='gray')
    axes[0].set_title('Raw (ground truth)')
    axes[0].axis('off')

    axes[1].imshow(target.cpu().squeeze(), cmap='gray')
    axes[1].set_title('Reco (target)')
    axes[1].axis('off')

    axes[2].imshow(x_fbp.cpu().squeeze(), cmap='gray')
    axes[2].set_title(f'FBP\nMSE={mse_fbp:.4f} | SSIM={ssim_fbp:.4f}')
    axes[2].axis('off')

    axes[3].imshow(x_rec.cpu().squeeze(), cmap='gray')
    axes[3].set_title(f'UNet\nMSE={mse_unet:.4f} | SSIM={ssim_unet:.4f}')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Training loop su tutti i numeri di angoli
# ---------------------------------------------------------------------------
for n_angles in ANGLE_CONFIGS:
    print(f'\n{"="*60}')
    print(f'Training con {n_angles} angoli')
    print(f'{"="*60}')

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = (
        get_dataloaders(data_root, n_angles, data_shape=DATA_SHAPE, batch_size=BATCH_SIZE)
    )

    K = get_ct_operator(n_angles=n_angles, img_shape=(DATA_SHAPE, DATA_SHAPE))

    torch.manual_seed(0)
    model = SimpleUNet(in_ch=1, out_ch=1, base_ch=32).to(device)

    weights_path = weights_dir / f'CTUNet_{n_angles}angles_{LOSS_NAME}.pth'

    train_loss_history, val_loss_history, val_ssim_history, val_psnr_history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        K=K,
        device=device,
        weights_path=weights_path,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        loss_name=LOSS_NAME,
        use_scheduler=True,   # <-- aggiungi
        eta_min=1e-6,         # <-- aggiungi (opzionale, default già impostato)
    )

    plot_loss_curves(train_loss_history, val_loss_history, val_ssim_history, val_psnr_history)

    # Carica i pesi migliori e visualizza un sample del test set
    best_model = SimpleUNet(in_ch=1, out_ch=1, base_ch=32)
    best_model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    best_model = best_model.to(device)

    visualize_sample(
        best_model, test_dataset, K, device, n_angles,
        title=f'Risultato dopo training | {n_angles} angoli | loss={LOSS_NAME}',
    )

# ---------------------------------------------------------------------------
# Inference su tutto il test set per tutti i numeri di angoli
# ---------------------------------------------------------------------------
print(f'\n{"="*60}')
print('Inference su tutto il test set')
print(f'{"="*60}')

for n_angles in ANGLE_CONFIGS:
    print(f'\n--- {n_angles} angoli | loss: {LOSS_NAME} ---')

    _, _, test_dataset, _, _, test_loader = get_dataloaders(
        data_root, n_angles, data_shape=DATA_SHAPE, batch_size=BATCH_SIZE
    )

    K = get_ct_operator(n_angles=n_angles, img_shape=(DATA_SHAPE, DATA_SHAPE))

    model = SimpleUNet(in_ch=1, out_ch=1, base_ch=32)
    weights_path = weights_dir / f'CTUNet_{n_angles}angles_{LOSS_NAME}.pth'
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model = model.to(device)
    model.eval()

    total_mse_fbp  = 0.0
    total_mse_unet = 0.0
    total_ssim_fbp  = 0.0
    total_ssim_unet = 0.0
    n_batches = 0

    with torch.no_grad():
        for x_batch, y_delta_batch, target_batch in test_loader:
            x_batch      = x_batch.to(device)
            y_delta_batch = y_delta_batch.to(device)
            target_batch  = target_batch.to(device)

            x_fbp = torch.clamp(K.FBP(y_delta_batch), min=0.0, max=1.0)
            x_rec = model(x_fbp)

            total_mse_fbp  += torch.mean((x_fbp - target_batch) ** 2).item()
            total_mse_unet += torch.mean((x_rec - target_batch) ** 2).item()
            total_ssim_fbp  += structural_similarity_index_measure(
                x_fbp, target_batch, data_range=1.0
            ).item()
            total_ssim_unet += structural_similarity_index_measure(
                x_rec, target_batch, data_range=1.0
            ).item()
            n_batches += 1

    n = max(n_batches, 1)
    print(f'  FBP  -> MSE: {total_mse_fbp  / n:.5f} | SSIM: {total_ssim_fbp  / n:.4f}')
    print(f'  UNet -> MSE: {total_mse_unet / n:.5f} | SSIM: {total_ssim_unet / n:.4f}')

    visualize_sample(
        model, test_dataset, K, device, n_angles,
        title=f'Inference finale | {n_angles} angoli | loss={LOSS_NAME}',
    )