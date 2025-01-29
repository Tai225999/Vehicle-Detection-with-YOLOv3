import torch
import numpy as np
import config

from collections import defaultdict
from matplotlib import pyplot as plt, patches


def iou_width_height(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Vectorized IoU computation for two sets of boxes based on width and height.
    Each box is in the format [width, height].

    Parameters:
        boxes1 (Tensor): Boxes of shape (N, 2), where each row is [width, height].
        boxes2 (Tensor): Boxes of shape (M, 2), where each row is [width, height].

    Returns:
        Tensor: IoU matrix of shape (N, M), where each element is the IoU between boxes1[i] and boxes2[j].
    """
    # Compute intersection area
    min_width = torch.min(boxes1[:, None, 0], boxes2[:, 0])  # Shape: (N, M)
    min_height = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # Shape: (N, M)
    intersection_area = min_width * min_height  # Shape: (N, M)

    # Compute areas of boxes1 and boxes2
    boxes1_area = boxes1[:, 0] * boxes1[:, 1]  # Shape: (N,)
    boxes2_area = boxes2[:, 0] * boxes2[:, 1]  # Shape: (M,)

    # Compute IoU
    iou = intersection_area / (boxes1_area[:, None] + boxes2_area - intersection_area + 1e-6)  # Shape: (N, M)

    return iou


def intersection_over_union(boxes_preds, boxes_labels, device="cuda"):
    """
    This function calculates intersection over union (IoU) given predicted boxes
    and target boxes. If the boxes are identical, it returns 1.0 for those boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)

    Returns:
        tensor: Intersection over union for all examples
    """
    # Convert to device
    boxes_preds = boxes_preds.to(device)
    boxes_labels = boxes_labels.to(device)

    # Check if the boxes are identical
    identical_boxes = torch.all(torch.isclose(boxes_preds, boxes_labels, atol=1e-6), dim=-1)

    # Calculate coordinates for predicted boxes
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

    # Calculate coordinates for ground truth boxes
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    # Calculate intersection coordinates
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Calculate intersection area
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Calculate areas of the boxes
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # Calculate IoU
    iou = intersection / (box1_area + box2_area - intersection + 1e-6)

    # If boxes are identical, return 1.0 for those boxes
    if torch.any(identical_boxes):
        # Set IoU to 1.0 for identical boxes
        iou[identical_boxes] = 1.0
    return iou


def non_max_suppression(bboxes, iou_threshold, threshold, device="cuda"):
    """
    Vectorized version of Non Max Suppression given bboxes

    Parameters:
        bboxes (tensor): tensor of shape (N, 6) where each row is [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        tensor: tensor of bboxes after performing NMS given a specific IoU threshold
    """
    # If no bboxes, return empty tensor
    if len(bboxes) == 0:
        return torch.tensor([])
    
    # Convert to device
    bboxes = bboxes.to(device)

    # Filter out boxes with confidence score below the threshold
    mask = bboxes[:, 1] > threshold
    bboxes = bboxes[mask.to(device)]

    # Sort boxes by confidence score in descending order
    _, sorted_indices = torch.sort(bboxes[:, 1], descending=True)
    bboxes = bboxes[sorted_indices]

    # Initialize an empty list to store the selected boxes after NMS
    bboxes_after_nms = []

    while bboxes.shape[0] > 0:
        # Select the box with the highest confidence score
        chosen_box = bboxes[0]
        bboxes_after_nms.append(chosen_box)

        # Remove the chosen box from the list
        bboxes = bboxes[1:]

        #print("bbox before: ", bboxes)

        if bboxes.shape[0] == 0:
            break

        # Keep the boxes that have different class than the chosen box
        mask1 = bboxes[:, 0] != chosen_box[0]

        #print("bbox1: ", mask1)

        # Compute IoU between the chosen box and the remaining boxes
        ious = intersection_over_union(
            chosen_box[2:].unsqueeze(0),
            bboxes[:, 2:],
        ).reshape(-1)

        # Remove the boxes that have IoU higher than the threshold
        mask2 = ious < iou_threshold

        #print("bbox2: ", mask2)

        #print("final mask:", mask1 | mask2)

        bboxes = bboxes[(mask1 | mask2).to(device)]

        #print("bbox after: ", bboxes)

    # Convert the list of selected boxes to a tensor
    if len(bboxes_after_nms) > 0:
        bboxes_after_nms = torch.stack(bboxes_after_nms)
    else:
        bboxes_after_nms = torch.tensor([])

    return bboxes_after_nms

def intersection_over_union_matrix(boxes1, boxes2):
    """
    Vectorized IoU computation for two sets of boxes.
    Parameters:
        boxes1 (Tensor): Boxes of shape (N, 4), where each row is (x1, y1, x2, y2).
        boxes2 (Tensor): Boxes of shape (M, 4), where each row is (x1, y1, x2, y2).
    Returns:
        Tensor: IoU matrix of shape (N, M), where each element is the IoU between boxes1[i] and boxes2[j].
    """
    # Get the coordinates of the intersection rectangle
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    # Compute the area of the intersection rectangle
    intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Compute the area of both sets of boxes
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute the IoU
    iou = intersection_area / (boxes1_area[:, None] + boxes2_area - intersection_area + 1e-6)

    return iou


def mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold: float = 0.5,
    num_classes: int = 6,
    device: str = "cuda",
) -> float:
    """
    Vectorized calculation of mean average precision (mAP) for object detection.
    Parameters:
        pred_boxes (Tensor): Predictions of Bounding Boxes (N, 7) [img_idx, class_pred, prob_score, x1, y1, x2, y2]
        true_boxes (Tensor): Correct labels of Bounding Boxes (N, 6) [img_idx, class_target, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        num_classes (int): number of classes
        device (str): device to perform calculations
    Returns:
        float: mAP
    """

    #If there are no detections, return 0
    if pred_boxes.shape[0] == 0:
        return 0.0
    # Move tensors to the specified device
    pred_boxes = pred_boxes.to(device)
    true_boxes = true_boxes.to(device)

    # List storing all AP for respective classes
    average_precisions = []

    # Used for numerical stability
    epsilon = 1e-6

    for c in range(num_classes):
        # Filter predictions and ground truths for the current class
        detections = pred_boxes[pred_boxes[:, 1] == c]
        ground_truths = true_boxes[true_boxes[:, 1] == c]

        # If there are no ground truths for this class, skip
        if ground_truths.shape[0] == 0:
            average_precisions.append(torch.tensor(0.0, device=device))
            continue

        # Sort detections by confidence score in descending order
        detections = detections[detections[:, 2].argsort(descending=True)]

        # Get unique image indices in ground truths
        image_indices = ground_truths[:, 0].unique()

        # Initialize True Positives and False Positives
        TP = torch.zeros(detections.shape[0], device=device)
        FP = torch.zeros(detections.shape[0], device=device)

        # Add a match flag column to ground_truths
        ground_truths = torch.cat(
            [ground_truths, torch.zeros(ground_truths.shape[0], 1, device=device)], dim=1
        )

        # Iterate over each image
        for img_idx in image_indices:
            img_ground_truths = ground_truths[ground_truths[:, 0] == img_idx]
            img_detections = detections[detections[:, 0] == img_idx]

            if img_detections.shape[0] == 0:
                continue

            # Compute IoU between all detections and ground truths in the image
            iou_matrix = intersection_over_union_matrix(img_detections[:, 3:], img_ground_truths[:, 3:7])

            # Find the best matching ground truth for each detection
            best_iou, best_gt_idx = iou_matrix.max(dim=1)

            # Mark detections as TP or FP
            for i, (iou, gt_idx) in enumerate(zip(best_iou, best_gt_idx)):
                if iou > iou_threshold:
                    if img_ground_truths[gt_idx, 7] == 0:  # Check if GT is unmatched (using the new flag column)
                        TP[i] = 1
                        img_ground_truths[gt_idx, 7] = 1  # Mark GT as matched
                    else:
                        FP[i] = 1  # Duplicate detection
                else:
                    FP[i] = 1  # False Positive

        # Compute cumulative TP and FP
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        # Compute precision and recall
        recalls = TP_cumsum / (ground_truths.shape[0] + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        # Compute AP using the precision-recall curve
        precisions = torch.cat((torch.tensor([1.0], device=device), precisions))
        recalls = torch.cat((torch.tensor([0.0], device=device), recalls))

        # Integrate under the precision-recall curve
        ap = torch.trapz(precisions, recalls)

        # Append AP for this class
        average_precisions.append(ap)

    # Compute mean Average Precision (mAP) across all classes
    mAP = torch.mean(torch.tensor(average_precisions, device=device))

    return mAP.item()


def plot_image(image
               , boxes: list):
    """
    Plots predicted bounding boxes on the image
    Parameters:
        image (np.ndarray): image of shape (H, W, C)
    """
    cmap = plt.get_cmap("tab20b")
    class_labels = config.CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()


def process_predictions(predictions, anchors, conf_threshold=0.5):
    """
    Convert model predictions to bounding box format.

    Args:
        predictions: Model predictions for a single scale.
        anchors: Anchor boxes for the scale.
        conf_threshold: Confidence threshold for filtering predictions.

    Returns:
        List of bounding boxes in the format [image_idx, class, confidence, x, y, w, h].
    """
    pred_boxes = []
    batch_size, _, grid_size, _ = predictions.shape

    for i in range(batch_size):
        for j in range(grid_size):
            for k in range(grid_size):
                # Get the prediction for this cell and anchor
                pred = predictions[i, :, j, k]

                # Extract confidence and class probabilities
                conf = torch.sigmoid(pred[0])
                class_probs = torch.softmax(pred[5:], dim=0)
                class_idx = torch.argmax(class_probs).item()

                # Filter out low-confidence predictions
                if conf < conf_threshold:
                    continue

                # Convert predictions to bounding box format
                x, y, w, h = pred[1:5]
                x = (x + k) / grid_size
                y = (y + j) / grid_size
                w = torch.exp(w) * anchors[0] / grid_size
                h = torch.exp(h) * anchors[1] / grid_size

                pred_boxes.append([i, class_idx, conf.item(), x.item(), y.item(), w.item(), h.item()])

    return pred_boxes


def convert_to_bboxes(predictions, anchors, S: int, is_preds: bool=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: tensor of converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(batch_size, num_anchors * S * S, 6).to(predictions.device)
    return converted_bboxes.tolist()

def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    device="cuda",
):
    """
    Given the model and the dataloader, this function returns the bboxes and the labels for each image.
    Parameters:
        loader: the DataLoader object
        model: the model
        iou_threshold: the threshold where predicted bboxes is correct
        anchors: the anchors used for the predictions
        threshold: the threshold to remove predicted bboxes (independent of IoU)
        device: the device to run the model on
    Returns:
        all_pred_boxes: a list containing all the predicted bboxes for each image
        all_true_boxes: a list containing all the ground truth bboxes for each image
    """
    # Make sure model is in eval mode
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]

        # Process predictions for each scale
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device)
            boxes_scale_i = convert_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, box in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # Convert ground truth labels to bboxes
        true_bboxes = convert_to_bboxes(
            labels[2].to(device), anchor, S=S, is_preds=False
        )

        # Process each image in the batch
        for idx in range(batch_size):
            # Concatenate predicted bboxes for the current image
            pred_boxes = torch.tensor(bboxes[idx]).to(device)
            truth_boxes = torch.tensor(true_bboxes[idx]).to(device)

            # Apply Non-Max Suppression (NMS) to predicted bboxes
            nms_boxes = non_max_suppression(
                pred_boxes,
                iou_threshold=iou_threshold,
                threshold=threshold,
            )
            
            nms_truth = non_max_suppression(
                truth_boxes,
                iou_threshold=iou_threshold,
                threshold=threshold
            )

            # Append predicted bboxes to the list
            if nms_boxes.shape[0] > 0:
                all_pred_boxes.append(
                    torch.cat(
                        [
                            torch.full((nms_boxes.shape[0], 1), train_idx, device=device),
                            nms_boxes,
                        ],
                        dim=1,
                    )
                )

            # Append ground truth bboxes to the list
            if nms_truth.shape[0] > 0:
                all_true_boxes.append(
                    torch.cat(
                        [
                            torch.full((nms_truth.shape[0], 1), train_idx, device=device),
                            nms_truth,
                        ],
                        dim=1,
                    )
                )

            train_idx += 1

    # Concatenate all predicted and ground truth bboxes
    all_pred_boxes = torch.cat(all_pred_boxes, dim=0).tolist() if all_pred_boxes else []
    all_true_boxes = torch.cat(all_true_boxes, dim=0).tolist() if all_true_boxes else []

    # Set model back to training mode
    model.train()
    return all_pred_boxes, all_true_boxes