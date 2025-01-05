CLASSES = ['bus', 'car', 'microbus', 'motorbike', 'pickup-van', 'tricycle', 'truck']

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
