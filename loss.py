import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        """
        Computes the loss for a particular scale in the YOLOv3 model.
        Args:
            predictions (torch.Tensor): Output from the model of shape 
                (batch size, anchors on scale, grid size, grid size, 5 + num classes).
            target (torch.Tensor): Targets on the particular scale of shape 
                (batch size, anchors on scale, grid size, grid size, 6).
            anchors (torch.Tensor): Anchor boxes on the particular scale of shape 
                (anchors on scale, 2).
        Returns:
            torch.Tensor: The computed loss for the particular scale.
        """

        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i
    
        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        # Convert predictions to bounding boxes
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        # Compute IoU between predictions and targets
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        # Object loss
        object_loss = self.bce((predictions[..., 0:1][obj]), ious)

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        # Apply sigmoid to x, y coordinates
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        # Transform targets for width and height
        target[..., 3:5] = torch.log(1e-16 + target[..., 3:5] / anchors)
        # Box loss
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        # Total loss
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )