"""
Training script for the SFRC19 detection SSD model
"""
import torch

import data_processing as dp


def adjust_anchors(bbox_preds, anchors_orig):
    """
    Adjusts the anchor boxes using the SSMD network predictions and reshapes for compatibility with bbox labels.
    Also forces any anchor boxes with limits outside of the image to be in the range 0 and 1.
    :param bbox_preds: Predicted bounding box locations -- has size (batch_size, # anchors/pixel * 4, image size, image_size)
    :param anchors_orig: Prior anchor boxes to adjust -- has size ( # anchors/pixel, 4, image size, image_size)
    :return: adjusted_anchors
    """

    # Format anchor boxes into same size as bbox_preds
    adjusted_anchors = []
    for i in range(len(anchors_orig)):
        # Reshape for convenience
        anchors_tmp = anchors_orig[i].unsqueeze(dim=0).repeat(bbox_preds[0].size()[0], 1, 1, 1, 1)
        bbox_preds_tmp = bbox_preds[i].unsqueeze(dim=0).reshape(bbox_preds[i].size()[0], anchors_orig[i].size()[0],
                                                                anchors_orig[i].size()[1], bbox_preds[i].size()[2],
                                                                bbox_preds[i].size()[3])

        # Define vales
        anchors_x = anchors_tmp[:, :, 0, :, :]
        anchors_y = anchors_tmp[:, :, 1, :, :]
        anchors_width = anchors_tmp[:, :, 2, :, :]
        anchors_height = anchors_tmp[:, :, 3, :, :]

        bbox_x = bbox_preds_tmp[:, :, 0, :, :]
        bbox_y = bbox_preds_tmp[:, :, 1, :, :]
        bbox_width = bbox_preds_tmp[:, :, 2, :, :]
        bbox_height = bbox_preds_tmp[:, :, 3, :, :]

        x_preds = (anchors_x + anchors_width * bbox_x).unsqueeze(dim=2)
        y_preds = (anchors_y + anchors_height * bbox_y).unsqueeze(dim=2)
        width_preds = (anchors_width * torch.exp(bbox_width)).unsqueeze(dim=2)
        height_preds = (anchors_height * torch.exp(bbox_height)).unsqueeze(dim=2)

        adjusted_anchors.append(torch.cat((x_preds, y_preds, width_preds, height_preds), dim=2))

        adjusted_anchors[i] = adjusted_anchors[i].unsqueeze(1).repeat(1, 8, 1, 1, 1, 1)

    return adjusted_anchors


def calculate_iou(bboxes, adjusted_anchors):
    """
    Calculates the Intersection over Union score to select the best boxes
    :param bboxes: Ground truth bounding box scores
    -- has size (batch_size, max_boxes, 5 [class_label, x, y, width, height])
    :param adjusted_anchors: Anchors boxes adjusted by the network predictions
    -- has size (batch_size, # anchors per pixel, 4, feature_size, feature_size
    :return: iou_tensor: torch tensor with IoU values
    """

    # Loop through adjusted anchor box list
    IoUs = []
    for i in range(len(adjusted_anchors)):
        # Get shape of bboxes compatible with adjusted anchors
        feature_size = adjusted_anchors[i].size()[-1]
        anchors_per_pixel = adjusted_anchors[i].size()[2]
        adjusted_bboxes = bboxes.unsqueeze(2).unsqueeze(4).unsqueeze(5).repeat(1,1,anchors_per_pixel
                                                                               ,1,feature_size,feature_size)

        # Define box and anchor parameters
        x_min = adjusted_anchors[i][:, :, :, 0, :, :]
        y_min = adjusted_anchors[i][:, :, :, 1, :, :]
        width = adjusted_anchors[i][:, :, :, 2, :, :]
        height = adjusted_anchors[i][:, :, :, 3, :, :]

        x_min_box = adjusted_bboxes[:, :, :, 1, :, :]
        y_min_box = adjusted_bboxes[:, :, :, 2, :, :]
        width_box = adjusted_bboxes[:, :, :, 3, :, :]
        height_box = adjusted_bboxes[:, :, :, 4, :, :]

        x_left = torch.max(x_min, x_min_box)
        y_top = torch.max(y_min, y_min_box)
        x_right = torch.min(x_min + width, x_min_box + width_box)
        y_bottom = torch.min(y_min + height, y_min_box + height_box)

        # Check if boxes overlap
        bool_tensor = torch.logical_or(x_right <= x_left, y_bottom <= y_top)  # Warnings aren't valid

        # Calculate intersection
        intersection = (x_right - x_left) * (y_bottom - y_top)
        intersection[bool_tensor] = 0.0  # If boxes don't overlap intersection is 0

        # Calculate Union
        anchor_area = width * height
        box_area = width_box * height_box
        union = (anchor_area + box_area) - intersection

        # Calculate IoU
        iou = intersection/union
        IoUs.append(iou)

    return IoUs


def non_maximal_suppression(adjusted_anchors, IoUs):
    """
    Uses non-maximal suppression to select which adjusted anchor to use from the adjusted_anchors tensor
    :param adjusted_anchors: list of torch tensors of adjusted anchor box values
    :param IoUs: list of torch tensors with Intersection over Union values
    :return: selected_anchors
    """


def train_ssmd(model, bbox_criterion, cls_criterion, optimizer, X, Y, epochs, batch_size):
    """
    Training function for the SSMD model to predict opacity and bounding boxes
    :param model: SSMD model to train
    :param bbox_criterion: The loss function used for bounding box prediction
    :param cls_criterion: The loss function used for opacity prediction
    :param optimizer: Optimizing algorithm to use
    :param X: Input images for training
    :param Y: Bounding box and opacity labels
    :param epochs: Number of epochs to train for
    :param batch_size: Batch size used in training
    :return: predictions
    """

    for epoch in range(epochs):
        running_loss = 0.0
        permutation = torch.randperm(X.size()[0])

        for i in range(0, X.size()[0], batch_size):
            # Zero out gradients
            optimizer.zero_grad()

            # Define batch
            indices = permutation[i:i+batch_size]
            X_batch, Y_batch = X[indices, :, :, :], Y[indices, :, :, :]

            # Pass data through network
            conf_scores, bbox_preds, anchors = model(X)

            # Calculate IoU between bbox_preds and actual boxes



if __name__ == '__main__':
    # Load data
    images, class_labels, bboxes, scales = dp.load_sfrc_data('/data/')
