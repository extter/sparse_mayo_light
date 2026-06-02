import torch
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
    peak_signal_noise_ratio,
)
from tqdm.auto import tqdm

from .losses import get_loss

from torch.optim.lr_scheduler import CosineAnnealingLR

def _print_tensor_stats(name: str, t: torch.Tensor):
    print(f'  {name:20s} | min={t.min():.4f}  max={t.max():.4f}  mean={t.mean():.4f}  std={t.std():.4f}')


def train_one_epoch(model, train_loader, optimizer, loss_fn, K, device, epoch, num_epochs):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [train]', leave=True)

    for step, (x_batch, y_delta_batch, target_batch) in enumerate(progress_bar, start=1):
        x_batch       = x_batch.to(device)
        y_delta_batch = y_delta_batch.to(device)
        target_batch  = target_batch.to(device)

        with torch.no_grad():
            x_fbp = K.FBP(y_delta_batch)

        # PRIMA: x_pred = model(x_fbp)
        # ORA: predici gli artefatti e sottrai
        x_art = model(x_fbp)          # artefatti predetti
        x_rec = x_fbp - x_art         # immagine corretta

        loss = loss_fn(x_rec, target_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(
            batch_loss=f'{loss.item():.6f}',
            avg_loss=f'{epoch_loss / step:.6f}',
        )

    return epoch_loss / len(train_loader)


def validate(model, val_loader, loss_fn, K, device, epoch, num_epochs):
    model.eval()
    val_loss = 0.0
    val_ssim = 0.0
    val_psnr = 0.0
    n_batches = 0

    with torch.no_grad():
        for x_batch, y_delta_batch, target_batch in tqdm(
            val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [val]', leave=False
        ):
            x_batch       = x_batch.to(device)
            y_delta_batch = y_delta_batch.to(device)
            target_batch  = target_batch.to(device)

            x_fbp = K.FBP(y_delta_batch)

            # artefatti + ricostruzione
            x_art = model(x_fbp)
            x_rec = x_fbp - x_art

            # loss sempre sull'immagine finale
            val_loss += loss_fn(x_rec, target_batch).item()

            # metriche sulla ricostruzione finale
            val_ssim += structural_similarity_index_measure(
                x_rec, target_batch, data_range=1.0
            ).item()
            val_psnr += peak_signal_noise_ratio(
                x_rec, target_batch, data_range=1.0
            ).item()
            n_batches += 1

    n = max(n_batches, 1)
    return val_loss / n, val_ssim / n, val_psnr / n

def train(
    model,
    train_loader,
    val_loader,
    K,
    device,
    weights_path,
    num_epochs=10,
    lr=1e-3,
    loss_name='mixed',
    use_scheduler=True,       # <-- nuovo
    eta_min=1e-6,             # <-- LR minimo a fine annealing
):
    loss_fn = get_loss(loss_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min) if use_scheduler else None

    train_loss_history = []
    val_loss_history = []
    val_ssim_history = []
    val_psnr_history = []
    best_ssim = -1.0

    print(f'Loss function: {loss_name}')
    if scheduler:
        print(f'Scheduler: CosineAnnealingLR (T_max={num_epochs}, eta_min={eta_min})')

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, K, device, epoch, num_epochs
        )
        val_loss, val_ssim, val_psnr = validate(
            model, val_loader, loss_fn, K, device, epoch, num_epochs
        )

        if scheduler:
            scheduler.step()

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_ssim_history.append(val_ssim)
        val_psnr_history.append(val_psnr)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch + 1}/{num_epochs} | '
            f'Train loss: {train_loss:.6f} | '
            f'Val loss: {val_loss:.6f} | '
            f'Val SSIM: {val_ssim:.4f} | '
            f'Val PSNR: {val_psnr:.2f} dB | '
            f'LR: {current_lr:.2e}'        # <-- utile per vedere il decay
        )

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), weights_path)

    print(f'Saved best model weights to: {weights_path}')
    return train_loss_history, val_loss_history, val_ssim_history, val_psnr_history