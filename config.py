import torch

CLASSES = ['bus', 'car', 'microbus', 'motorbike', 'pickup-van', 'tricycle', 'truck']

NUM_CLASSES = len(CLASSES)

DATA_DIR = "Dataset"

TRAIN_DIR = "train"

VALID_DIR = "valid"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 2e-5

ANCHORS = [
    [(0.66, 0.68), (0.40, 0.32), (0.20, 0.39)],
    [(0.21, 0.19), (0.12, 0.20), (0.10, 0.12)],
    [(0.06, 0.10), (0.04, 0.06), (0.02, 0.04)]
]

ARCHITECTURE = [
    #This is the architecture of the YOLOv3 model.
    #A tuple represents a convolutional layer in the format (filters, kernel_size, stride).
    #A list represents a Residual Block, with the number of repeats.
    #"S" represents a scale prediction layer.
    #"U" represents an upsampling layer.    
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    # first route from the end of the previous block
    (512, 3, 2),
    ["B", 8], 
    # second route from the end of the previous block
    (1024, 3, 2),
    ["B", 4],
    # until here is YOLO-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]
