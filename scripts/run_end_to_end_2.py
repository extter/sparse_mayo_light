"""
Training end-to-end per tutti e 4 i numeri di angoli,
inference sul test set con metriche per slice (SSIM, PSNR, MSE),
salvataggio CSV e boxplot comparativi.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
    peak_signal_noise_ratio,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from methods.end_to_end_2 import (
    ResidualUNet,
    SimpleUNet,
    get_dataloaders,
    get_ct_operator,
    get_device,
    plot_loss_curves,
    train,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
book_root   = Path(__file__).resolve().parents[1]
data_root   = (book_root / 'data').resolve()
weights_dir = book_root / 'methods' / 'end_to_end_2' / 'weights'
eval_results_dir = book_root / 'evaluation' / 'results'
results_dir = eval_results_dir / 'end_to_end_2'
weights_dir.mkdir(exist_ok=True)
eval_results_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

device = get_device()

ANGLE_CONFIGS = [45, 60, 90, 180]
NUM_EPOCHS    = 10
BATCH_SIZE    = 16
LR            = 1e-3
DATA_SHAPE    = 256
LOSS_NAME     = 'mixed'   # 'mse' | 'ssim' | 'ssim_custom' | 'mixed'

print('Device     :', device)
print('Data root  :', data_root)
print('Weights dir:', weights_dir)
print('Results dir:', results_dir)
print('Loss       :', LOSS_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics_per_slice(pred: torch.Tensor,
                               target: torch.Tensor) -> dict:
    """
    Calcola SSIM, PSNR e MSE per ogni slice nel batch.
    pred/target: [B, 1, H, W] float, range [0, 1].
    Ritorna dict con liste di lunghezza B.
    """
    B = pred.shape[0]
    ssims, psnrs, mses = [], [], []
    for i in range(B):
        p = pred[i].unsqueeze(0).clamp(0.0, 1.0)   # [1,1,H,W]
        t = target[i].unsqueeze(0)
        ssims.append(structural_similarity_index_measure(p, t, data_range=1.0).item())
        psnrs.append(peak_signal_noise_ratio(p, t, data_range=1.0).item())
        mses.append(torch.mean((p - t) ** 2).item())
    return {'ssim': ssims, 'psnr': psnrs, 'mse': mses}


def visualize_sample(model, dataset, K, device, n_angles, title=''):
    model.eval()
    x_true, y_delta, target = dataset[0]
    x_true  = x_true.unsqueeze(0).to(device)
    y_delta = y_delta.unsqueeze(0).to(device)
    target  = target.unsqueeze(0).to(device)

    with torch.no_grad():
        x_fbp = K.FBP(y_delta)
        x_art = model(x_fbp)
        x_rec = (x_fbp - x_art).clamp(0.0, 1.0)

    mse_fbp   = torch.mean((x_fbp - target) ** 2).item()
    mse_unet  = torch.mean((x_rec  - target) ** 2).item()
    ssim_fbp  = structural_similarity_index_measure(x_fbp, target, data_range=1.0).item()
    ssim_unet = structural_similarity_index_measure(x_rec,  target, data_range=1.0).item()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(title, fontsize=13)
    for ax, img, t in zip(
        axes,
        [x_true, target, x_fbp, x_rec],
        ['Raw (ground truth)', 'Reco (target)',
         f'FBP\nMSE={mse_fbp:.4f} | SSIM={ssim_fbp:.4f}',
         f'UNet\nMSE={mse_unet:.4f} | SSIM={ssim_unet:.4f}'],
    ):
        ax.imshow(img.cpu().squeeze(), cmap='gray')
        ax.set_title(t)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for n_angles in ANGLE_CONFIGS:
    print(f'\n{"="*60}')
    print(f'Training con {n_angles} angoli')
    print(f'{"="*60}')

    train_ds, val_ds, test_ds, train_loader, val_loader, _ = get_dataloaders(
        data_root, n_angles, data_shape=DATA_SHAPE, batch_size=BATCH_SIZE
    )
    K = get_ct_operator(n_angles=n_angles, img_shape=(DATA_SHAPE, DATA_SHAPE))

    torch.manual_seed(0)
    model = SimpleUNet(in_ch=1, out_ch=1, base_ch=32).to(device)
    weights_path = weights_dir / f'CTUNet_{n_angles}angles_{LOSS_NAME}.pth'

    train_loss_hist, val_loss_hist, val_ssim_hist, val_psnr_hist = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        K=K,
        device=device,
        weights_path=weights_path,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        loss_name=LOSS_NAME,
        use_scheduler=True,
        eta_min=1e-6,
    )

    plot_loss_curves(train_loss_hist, val_loss_hist, val_ssim_hist, val_psnr_hist)

    best = SimpleUNet(in_ch=1, out_ch=1, base_ch=32)
    best.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    best = best.to(device)
    visualize_sample(best, test_ds, K, device, n_angles,
                     title=f'Sample post-training | {n_angles} angoli | {LOSS_NAME}')


# ---------------------------------------------------------------------------
# Inference con metriche per slice
# ---------------------------------------------------------------------------
print(f'\n{"="*60}')
print('Inference su tutto il test set — metriche per slice')
print(f'{"="*60}')

all_records = []   # ogni elemento → una riga del CSV

for n_angles in ANGLE_CONFIGS:
    print(f'\n--- {n_angles} angoli ---')

    _, _, test_ds, _, _, test_loader = get_dataloaders(
        data_root, n_angles, data_shape=DATA_SHAPE, batch_size=BATCH_SIZE, verbose=False
    )
    K = get_ct_operator(n_angles=n_angles, img_shape=(DATA_SHAPE, DATA_SHAPE))

    model = SimpleUNet(in_ch=1, out_ch=1, base_ch=32)
    weights_path = weights_dir / f'CTUNet_{n_angles}angles_{LOSS_NAME}.pth'
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for x_batch, y_delta_batch, target_batch in test_loader:
            x_batch       = x_batch.to(device)
            y_delta_batch = y_delta_batch.to(device)
            target_batch  = target_batch.to(device)

            x_fbp = K.FBP(y_delta_batch)
            x_art = model(x_fbp)
            x_rec = (x_fbp - x_art).clamp(0.0, 1.0)

            fbp_m  = compute_metrics_per_slice(x_fbp, target_batch)
            unet_m = compute_metrics_per_slice(x_rec, target_batch)

            B = x_batch.shape[0]
            for i in range(B):
                all_records.append({
                    'n_angles'  : n_angles,
                    # FBP
                    'fbp_ssim'  : fbp_m['ssim'][i],
                    'fbp_psnr'  : fbp_m['psnr'][i],
                    'fbp_mse'   : fbp_m['mse'][i],
                    # UNet
                    'unet_ssim' : unet_m['ssim'][i],
                    'unet_psnr' : unet_m['psnr'][i],
                    'unet_mse'  : unet_m['mse'][i],
                })

    # Stampa medie
    df_tmp = pd.DataFrame([r for r in all_records if r['n_angles'] == n_angles])
    print(f"  FBP  -> SSIM: {df_tmp['fbp_ssim'].mean():.4f} | "
          f"PSNR: {df_tmp['fbp_psnr'].mean():.2f} dB | "
          f"MSE: {df_tmp['fbp_mse'].mean():.5f}")
    print(f"  UNet -> SSIM: {df_tmp['unet_ssim'].mean():.4f} | "
          f"PSNR: {df_tmp['unet_psnr'].mean():.2f} dB | "
          f"MSE: {df_tmp['unet_mse'].mean():.5f}")

    visualize_sample(model, test_ds, K, device, n_angles,
                     title=f'Inference finale | {n_angles} angoli | {LOSS_NAME}')

# ---------------------------------------------------------------------------
# Salva CSV
# ---------------------------------------------------------------------------
df = pd.DataFrame(all_records)
csv_path = results_dir / f'metrics_unet_{LOSS_NAME}.csv'
df.to_csv(csv_path, index=False)
print(f'\nMetriche salvate in: {csv_path}')

# ---------------------------------------------------------------------------
# Boxplot comparativi
# ---------------------------------------------------------------------------
METRICS = [
    ('ssim', 'SSIM',      'higher is better'),
    ('psnr', 'PSNR (dB)', 'higher is better'),
    ('mse',  'MSE',       'lower is better'),
]

COLORS = {
    'FBP' : '#4C9BE8',
    'UNet': '#E8824C',
}

for metric_key, metric_label, direction in METRICS:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.suptitle(f'{metric_label} — FBP vs UNet ({direction})', fontsize=14, fontweight='bold')

    for ax, n_angles in zip(axes, ANGLE_CONFIGS):
        sub = df[df['n_angles'] == n_angles]
        data_fbp  = sub[f'fbp_{metric_key}'].values
        data_unet = sub[f'unet_{metric_key}'].values

        bp = ax.boxplot(
            [data_fbp, data_unet],
            labels=['FBP', 'UNet'],
            patch_artist=True,
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker='o', markersize=3, linestyle='none', alpha=0.5),
            widths=0.5,
        )
        for patch, key in zip(bp['boxes'], ['FBP', 'UNet']):
            patch.set_facecolor(COLORS[key])
            patch.set_alpha(0.75)

        # Aggiungi valore mediano come testo
        for i, data in enumerate([data_fbp, data_unet], start=1):
            med = np.median(data)
            ax.text(i, med, f'{med:.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='black')

        ax.set_title(f'{n_angles} angoli\n(n={len(data_fbp)} slice)', fontsize=11)
        ax.set_xlabel('Metodo')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel(metric_label, fontsize=11)
    plt.tight_layout()

    plot_path = results_dir / f'boxplot_{metric_key}_{LOSS_NAME}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'Boxplot salvato: {plot_path}')
    plt.show()

# ---------------------------------------------------------------------------
# Tabella riassuntiva (media ± std per ogni configurazione)
# ---------------------------------------------------------------------------
print('\n' + '='*70)
print('TABELLA RIASSUNTIVA (mean ± std sul test set)')
print('='*70)

summary_rows = []
for n_angles in ANGLE_CONFIGS:
    sub = df[df['n_angles'] == n_angles]
    for method, prefix in [('FBP', 'fbp'), ('UNet', 'unet')]:
        row = {'angles': n_angles, 'method': method}
        for mk in ['ssim', 'psnr', 'mse']:
            vals = sub[f'{prefix}_{mk}']
            row[f'{mk}_mean'] = vals.mean()
            row[f'{mk}_std']  = vals.std()
        summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

# Stampa formattata
print(f"\n{'Angles':>7} {'Method':>6} | {'SSIM':>16} {'PSNR':>16} {'MSE':>16}")
print('-' * 70)
for _, r in summary.iterrows():
    print(
        f"{int(r['angles']):>7} {r['method']:>6} | "
        f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}   "
        f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f} dB   "
        f"{r['mse_mean']:.5f} ± {r['mse_std']:.5f}"
    )

summary_path = results_dir / f'summary_{LOSS_NAME}.csv'
summary.to_csv(summary_path, index=False)
print(f'\nTabella riassuntiva salvata in: {summary_path}')