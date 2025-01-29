import os
import torch
import config

from utils import mean_average_precision, get_evaluation_bboxes


def train_fn(
    train_loader, val_loader, model, optimizer, loss_fn, scaled_anchors,
    num_epochs, device, checkpoint_dir, scheduler=None
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    history = {'train_loss': [], 'val_loss': [], 'lr_history': [], 'val_map': []}
    best_val_loss = float('inf')
    warmup_epochs = 2
    initial_lr = optimizer.param_groups[0]["lr"]
    warmup_lr = initial_lr / 10
    epochs_no_improve = 0
    early_stopping_patience = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Warmup learning rate
        if epoch < warmup_epochs:  # Fixed condition from <= to <
            lr = warmup_lr + (initial_lr - warmup_lr) * (epoch / (warmup_epochs - 1))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            # LR will be handled by scheduler after validation
            pass

        print(f"Epoch {epoch+1}: Current Learning Rate: {lr:.6f}")

        # Training loop (without mixed precision)
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y0, y1, y2 = y[0].to(device), y[1].to(device), y[2].to(device)

            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0], device)
                + loss_fn(out[1], y1, scaled_anchors[1], device)
                + loss_fn(out[2], y2, scaled_anchors[2], device)
            )

            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y0, y1, y2 = y[0].to(device), y[1].to(device), y[2].to(device)
                out = model(x)
                loss = (
                    loss_fn(out[0], y0, scaled_anchors[0], device)
                    + loss_fn(out[1], y1, scaled_anchors[1], device)
                    + loss_fn(out[2], y2, scaled_anchors[2], device)
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        # Calculate validation mAP
        val_pred_boxes, val_true_boxes = get_evaluation_bboxes(
            val_loader, model, iou_threshold=0.5, anchors=scaled_anchors.tolist(),
            threshold=0.6, device=device
        )
        val_map = mean_average_precision(
            torch.tensor(val_pred_boxes), torch.tensor(val_true_boxes), num_classes=config.NUM_CLASSES
        )
        history['val_map'].append(val_map)

        # Record current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        history['lr_history'].append(current_lr)

        # Step scheduler after validation (if not in warmup)
        if epoch >= warmup_epochs and scheduler is not None:
            scheduler.step(val_loss)  # Use current epoch's validation loss

        # Save best model and checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mAP: {val_map:.4f}")

    return history