import os
import torch
import json
import pickle
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import config

from torch.utils.data import DataLoader
from model import YOLOv3
from dataset import YOLODataset
from loss import YoloLoss
from train import train_fn


# Initialize model
model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

# Load pretrained weights (excluding prediction heads)
checkpoint = torch.load(config.PRETRAIN_MODEL, map_location=config.DEVICE)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and 'pred' not in k}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# Define optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, verbose=True
)
loss_fn = YoloLoss()

# Correct anchor scaling based on feature map strides (not grid sizes)
scaled_anchors = torch.tensor(config.ANCHORS) / (1 / torch.tensor([13, 26, 52]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))

# DataLoaders
train_loader = DataLoader(
    YOLODataset(
        config.DATA_DIR + '/train.csv',
        config.DATA_DIR + '/train/images',
        config.DATA_DIR + '/train/labels',
        S=[13, 26, 52]
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    YOLODataset(
        config.DATA_DIR + '/valid.csv',
        config.DATA_DIR + '/valid/images',
        config.DATA_DIR + '/valid/labels',
        S=[13, 26, 52]
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=False
)

# Start training
history = train_fn(
    train_loader=train_loader,
    val_loader=valid_loader,
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    scaled_anchors=scaled_anchors,
    num_epochs=30,
    device=config.DEVICE,
    checkpoint_dir=os.path.join(config.C, "Checkpoints"),
    scheduler=scheduler  # Pass scheduler to train_fn
)

# Save as JSON
json_path = os.path.join("/kaggle/working/Checkpoints", "training_history.json")
with open(json_path, 'w') as f:
    json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=4)

# Save as pickle
pkl_path = os.path.join("/kaggle/working/Checkpoints", "training_history.pkl")
with open(pkl_path, 'wb') as f:
    pickle.dump(history, f)

print(f"Training history saved to {json_path} and {pkl_path}")