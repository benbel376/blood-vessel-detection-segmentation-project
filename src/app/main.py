import time
import torch
from torch.utils.data import DataLoader
from dataset import DriveDataset
from model import build_unet
from losses import DiceBCELoss
from train import train, evaluate
from utils import seeding, create_dir, epoch_time, calculate_metrics

# Configuration
H, W = 512, 512
batch_size = 2
lr = 1e-4
num_epochs = 50
checkpoint_path = "files/checkpoint.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
seeding(42)

# Create directory for saving models
create_dir("files")

# Load datasets
train_x = sorted(glob("path/to/train/images/*"))
train_y = sorted(glob("path/to/train/masks/*"))
valid_x = sorted(glob("path/to/valid/images/*"))
valid_y = sorted(glob("path/to/valid/masks/*"))

train_dataset = DriveDataset(train_x, train_y)
valid_dataset = DriveDataset(valid_x, valid_y)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Initialize model, optimizer, scheduler, and loss function
model = build_unet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
loss_fn = DiceBCELoss()

# Training loop
best_valid_loss = float("inf")
for epoch in range(num_epochs):
    start_time = time.time()
    
    train_loss = train(model, train_loader, optimizer, loss_fn, device)
    valid_loss = evaluate(model, valid_loader, loss_fn, device)

    if valid_loss < best_valid_loss:
        torch.save(model.state_dict(), checkpoint_path)
        best_valid_loss = valid_loss
        print(f"Validation loss improved, saving model checkpoint to {checkpoint_path}")

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f"Epoch: {epoch+1:02}, Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f}, Validation Loss: {valid_loss:.3f}")

# Load the best model and calculate metrics on test data
test_x = sorted(glob("path/to/test/images/*"))
test_y = sorted(glob("path/to/test/masks/*"))
test_dataset = DriveDataset(test_x, test_y)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2)

model.load_state_dict(torch.load(checkpoint_path))
model.eval()
metrics = [0.0] * 5
for i, (x, y) in enumerate(test_loader):
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_y = (pred_y > 0.5).float()

        metrics = list(map(lambda m, s: m + s, metrics, calculate_metrics(y, pred_y)))

print("Average metrics across test set:")
metrics = [m / len(test_loader) for m in metrics]
print(f"Jaccard: {metrics[0]:.4f}, F1: {metrics[1]:.4f}, Recall: {metrics[2]:.4f}, Precision: {metrics[3]:.4f}, Accuracy: {metrics[4]:.4f}")
