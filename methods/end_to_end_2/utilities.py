import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from third_party.IPPy_prof import operators, utilities as ippyutils


def get_device():
    return ippyutils.get_device()


def get_ct_operator(n_angles: int, img_shape=(256, 256), det_size=512):
    return operators.CTProjector(
        img_shape=img_shape,
        angles=np.linspace(0, np.pi, n_angles, endpoint=False),
        det_size=det_size,
        geometry='parallel',
    )


def plot_sample(x, target, y_delta, x_fbp, title_fbp='FBP'):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(x.cpu().squeeze(), cmap='gray')
    plt.title('Input image $x$')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(target.cpu().squeeze(), cmap='gray')
    plt.title('Ground truth reco')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(y_delta.cpu().squeeze(), cmap='gray', aspect='auto')
    plt.title('Corrupted sinogram $y^\\delta$')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(x_fbp.cpu().squeeze(), cmap='gray')
    plt.title(title_fbp)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_results(x_true, target, x_fbp, x_rec, mse_fbp, mse_unet, ssim_fbp, ssim_unet):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(x_true.cpu().squeeze(), cmap='gray')
    plt.title('Input raw')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(target.cpu().squeeze(), cmap='gray')
    plt.title('Ground truth reco')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(x_fbp.cpu().squeeze(), cmap='gray')
    plt.title(f'FBP MSE: {mse_fbp:.5f} | SSIM: {ssim_fbp:.4f}')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(x_rec.cpu().squeeze(), cmap='gray')
    plt.title(f'UNet MSE: {mse_unet:.5f} | SSIM: {ssim_unet:.4f}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_loss_curves(train_loss_history, val_loss_history, val_ssim_history, val_psnr_history=None):
    fig, axes = plt.subplots(1, 3 if val_psnr_history else 2, figsize=(15 if val_psnr_history else 10, 4))

    axes[0].plot(train_loss_history, label='Train loss')
    axes[0].plot(val_loss_history, label='Val loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss per epoch')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(val_ssim_history, label='Val SSIM', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM per epoch')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if val_psnr_history:
        axes[2].plot(val_psnr_history, label='Val PSNR', color='orange')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('PSNR (dB)')
        axes[2].set_title('PSNR per epoch')
        axes[2].grid(alpha=0.3)
        axes[2].legend()

    plt.tight_layout()
    plt.show()