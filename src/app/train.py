import torch

def train(model, loader, optimizer, loss_fn, device):
    """
    Training loop for one epoch.

    Parameters:
        model: The U-Net model.
        loader: Training data loader.
        optimizer: Optimizer for model training.
        loss_fn: Loss function.
        device: Device to use for computation.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    epoch_loss = 0.0
    
    for x, y in loader:
        x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def evaluate(model, loader, loss_fn, device):
    """
    Evaluation loop for one epoch.

    Parameters:
        model: The U-Net model.
        loader: Validation data loader.
        loss_fn: Loss function.
        device: Device to use for computation.

    Returns:
        Average validation loss for the epoch.
    """
    model.eval()
    epoch_loss = 0.0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)
