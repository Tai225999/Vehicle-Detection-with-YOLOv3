import torch
import os

SEED = 310704

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ANCHORS = [
    [(0.66, 0.69), (0.40, 0.32), (0.21, 0.35)],
    [(0.19, 0.18), (0.11, 0.18), (0.09, 0.11)],
    [(0.05, 0.09), (0.04, 0.05), (0.02, 0.04)]
]

CLASSES = ['bus', 'car', 'microbus', 'motorbike', 'pickup-van', 'truck']

NUM_CLASSES = len(CLASSES)

BATCH_SIZE = 16

ROOT_DIR = './'

print(os.listdir(ROOT_DIR))

DATA_DIR = os.path.join(ROOT_DIR, 'Dataset')

TRAIN_DIR = os.path.join(DATA_DIR,'train')

VALID_DIR = os.path.join(DATA_DIR,'valid')

TEST_DIR = os.path.join(DATA_DIR,'test')

TRAIN_CSV = os.path.join(DATA_DIR,'train.csv')

VALID_CSV = os.path.join(DATA_DIR,'valid.csv')

TEST_CSV = os.path.join(DATA_DIR,'test.csv')

CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'Checkpoints')

IMAGE_SIZE = 416

S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

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

PRETRAIN_MODEL = os.path.join(DATA_DIR, 'Pretrained Weights/78.1map_0.2threshold_PASCAL.tar')

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4