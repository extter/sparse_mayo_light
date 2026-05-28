import torch
from torchmetrics.functional.image import structural_similarity_index_measure
from tqdm.auto import tqdm

from .losses import get_loss


def train_one_epoch(model, train_loader, optimizer, loss_fn, K, device, epoch, num_epochs):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [train]', leave=True)

    for step, (x_batch, y_delta_batch, target_batch) in enumerate(progress_bar, start=1):
        x_batch = x_batch.to(device)
        y_delta_batch = y_delta_batch.to(device)
        target_batch = target_batch.to(device)

        with torch.no_grad():
            x_fbp = K.FBP(y_delta_batch)

        x_pred = model(x_fbp)
        loss = loss_fn(x_pred, target_batch)

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
    n_batches = 0

    with torch.no_grad():
        for x_batch, y_delta_batch, target_batch in tqdm(
            val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [val]', leave=False
        ):
            x_batch = x_batch.to(device)
            y_delta_batch = y_delta_batch.to(device)
            target_batch = target_batch.to(device)

            x_fbp = K.FBP(y_delta_batch)
            x_pred = model(x_fbp)

            val_loss += loss_fn(x_pred, target_batch).item()
            val_ssim += structural_similarity_index_measure(
                x_pred, target_batch, data_range=1.0
            ).item()
            n_batches += 1

    val_loss /= max(n_batches, 1)
    val_ssim /= max(n_batches, 1)
    return val_loss, val_ssim


def train(
    model,
    train_loader,
    val_loader,
    K,
    device,
    weights_path,
    num_epochs=10,
    lr=1e-3,
    loss_name='mixed',   # 'mse' | 'ssim' | 'ssim_custom' | 'mixed'
):
    loss_fn = get_loss(loss_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    val_loss_history = []
    val_ssim_history = []
    best_ssim = -1.0

    print(f'Loss function: {loss_name}')

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, K, device, epoch, num_epochs
        )
        val_loss, val_ssim = validate(
            model, val_loader, loss_fn, K, device, epoch, num_epochs
        )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_ssim_history.append(val_ssim)

        print(
            f'Epoch {epoch + 1}/{num_epochs} | '
            f'Train loss: {train_loss:.6f} | '
            f'Val loss: {val_loss:.6f} | '
            f'Val SSIM: {val_ssim:.4f}'
        )

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), weights_path)

    print(f'Saved best model weights to: {weights_path}')
    return train_loss_history, val_loss_history, val_ssim_history