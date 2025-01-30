import json
import os
import config
import matplotlib.pyplot as plt

# Load the history dictionary
with open(os.path.join(config.CHECKPOINT_DIR, "training_history.json")) as f:
    history = json.load(f)

def draw_graph(history):
    """
        Draw the graph of the training loss, validation loss, validation mAP, and learning rate.
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Training loss
    axes[0, 0].plot(history["train_loss"], label="train_loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Validation loss
    axes[0, 1].plot(history["val_loss"], label="val_loss", color="r")
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    # Validation mAP
    axes[1, 0].plot(history["val_map"], label="val_map", color="g")
    axes[1, 0].set_title("Validation mAP")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("mAP")
    axes[1, 0].legend()

    # Learning rate
    axes[1, 1].plot(history["lr_history"], label="lr", color="b")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

# Draw the graph
draw_graph(history)