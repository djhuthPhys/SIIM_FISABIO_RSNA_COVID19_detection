"""
Training script for the SFRC19 detection SSD model
"""
import torch

import torch.nn as nn
import data_processing as dp
import sfrc19_ssmd_model as ssm


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

        adjusted_anchors_tmp = torch.cat((x_preds, y_preds, width_preds, height_preds), dim=2)\
            .unsqueeze(1).repeat(1, 8, 1, 1, 1, 1)

        adjusted_anchors.append(adjusted_anchors_tmp)

    return adjusted_anchors


def process_anchors(bbox_labels, anchors_orig):
    """
    Calculates the Intersection over Union score to select the best boxes and calculates the bounding box offsets
    :param bbox_labels: Ground truth bounding box scores
    -- has size (batch_size, max_boxes, 5 [class_label, x, y, width, height])
    :param anchors_orig: Anchors boxes adjusted by the network predictions
    -- has size (batch_size, max boxes, # anchors per pixel, 4, feature_size, feature_size)
    :return: iou: torch tensor with IoU values
    """

    # Get shape of bboxes compatible with adjusted anchors
    feature_size = anchors_orig.size()[-1]
    anchors_per_pixel = anchors_orig.size()[0]
    anchors_tmp = anchors_orig.unsqueeze(dim=0).repeat(bbox_labels.size()[0], 1, 1, 1, 1).unsqueeze(1).repeat(1, 8, 1,
                                                                                                              1, 1, 1)
    adjusted_bboxes = bbox_labels.unsqueeze(2).unsqueeze(4).unsqueeze(5).repeat(1, 1, anchors_per_pixel,
                                                                                1, feature_size, feature_size)

    # Define box and anchor parameters
    x_min = anchors_tmp[:, :, :, 0, :, :]
    y_min = anchors_tmp[:, :, :, 1, :, :]
    width = anchors_tmp[:, :, :, 2, :, :]
    height = anchors_tmp[:, :, :, 3, :, :]

    x_min_box = adjusted_bboxes[:, :, :, 1, :, :]
    y_min_box = adjusted_bboxes[:, :, :, 2, :, :]
    width_box = adjusted_bboxes[:, :, :, 3, :, :]
    height_box = adjusted_bboxes[:, :, :, 4, :, :]

    x_left = torch.max(x_min, x_min_box)
    y_top = torch.max(y_min, y_min_box)
    x_right = torch.min(x_min + width, x_min_box + width_box)
    y_bottom = torch.min(y_min + height, y_min_box + height_box)

    # Check if boxes overlap
    bool_tensor = torch.logical_or(x_right <= x_left, y_bottom <= y_top)

    # Calculate intersection
    intersection = (x_right - x_left) * (y_bottom - y_top)
    intersection[bool_tensor] = 0.0  # If boxes don't overlap intersection is 0

    # Calculate Union
    anchor_area = width * height
    box_area = width_box * height_box
    union = (anchor_area + box_area) - intersection

    # Calculate IoU
    iou = intersection/union

    # Calculate anchor box offsets
    x_min_off = (x_min - x_min_box).unsqueeze(dim=3)
    y_min_off = (y_min - y_min_box).unsqueeze(dim=3)
    width_off = (width - width_box).unsqueeze(dim=3)
    height_off = (height - height_box).unsqueeze(dim=3)
    print(x_min_off.size())
    offsets = torch.cat((x_min_off, y_min_off, width_off, height_off), dim=3)


    return iou, offsets, adjusted_bboxes


def get_positive_anchors(bbox_labels, anchors_orig):
    """
    Labels the anchor boxes with IoUs over a threshold of 0.5 as positive (1) and under a threshold of 0.5 as
    negative (0), returns the anchor box offsets needed to match bounding box labels
    :param bbox_labels: List of ground truth bounding box tensors
    :param anchors_orig: list of default anchor box tensors
    :return: anchor_status
    """
    threshold = 0.5
    iou_list = []
    anchor_status = []
    anchor_offsets = []
    adjusted_bboxes = []
    for i in range(len(anchors_orig)):
        iou, offsets, new_bboxes = process_anchors(bbox_labels, anchors_orig[i])
        iou_list.append(iou)

        # Format iou to match bbox_labels and adjusted_anchors
        iou = iou.unsqueeze(3).repeat(1,1,1,4,1,1)
        anchor_status.append(iou > threshold)
        adjusted_bboxes.append(new_bboxes)
        anchor_offsets.append(offsets)

    return anchor_status, anchor_offsets, adjusted_bboxes


# def non_maximal_suppression(adjusted_anchors, IoUs):
#     """
#     Uses non-maximal suppression to select which adjusted anchor to use from the adjusted_anchors tensor
#     :param adjusted_anchors: list of torch tensors of adjusted anchor box values
#     :param IoUs: list of torch tensors with Intersection over Union values
#     :return: selected_anchors
#     """


def train_ssmd(model, bbox_criterion, cls_criterion, optimization, X, Y, epochs, batch_size):
    """
    Training function for the SSMD model to predict opacity and bounding boxes
    :param model: SSMD model to train
    :param bbox_criterion: The loss function used for bounding box prediction
    :param cls_criterion: The loss function used for opacity prediction
    :param optimization: Optimizing algorithm to use
    :param X: Input images for training
    :param Y: Bounding box and opacity labels
    :param epochs: Number of epochs to train for
    :param batch_size: Batch size used in training
    :return: predictions
    """
    max_boxes = Y.size()[1]

    for epoch in range(epochs):
        running_loss = 0.0
        permutation = torch.randperm(X.size()[0])

        for i in range(0, X.size()[0], batch_size):
            # Zero out gradients
            optimization.zero_grad()

            # Define batch
            indices = permutation[i:i+batch_size]
            X_batch, Y_batch = X[indices, :, :, :], Y[indices, :, :, :]

            # Pass data through network
            conf_scores, bbox_preds, anchors = model(X)

            # Reshape bbox_preds for localization loss calculation
            shaped_bbox_preds = []
            for k in range(len(bbox_preds)):
                anchors_per_pixel = int(bbox_preds[k].size()[1]/4)
                feature_size = bbox_preds[k].size()[-1]
                shaped_bbox_preds.append(bbox_preds[k].reshape((batch_size, anchors_per_pixel, 4, feature_size,
                                                                feature_size)).unsqueeze(1).repeat(1,8,1,1,1,1))

            # Determine positive and negative boxes/anchors and calculate anchor offsets
            anchor_status, anchor_offsets, adjusted_bboxes = get_positive_anchors(Y, anchors)
            bbox_coords = adjusted_bboxes[:,:,:,1:-1,:,:]
            positive_bboxes = []
            positive_anchors = []
            positive_offsets = []
            positive_bbox_preds = []
            for j in range(len(anchors)):
                positive_anchors.append(anchors[j][anchor_status[j]])
                positive_bboxes.append(bbox_coords[j][anchor_status[j]])
                positive_offsets.append(anchor_offsets[j][anchor_status[j]])
                positive_bbox_preds.append(shaped_bbox_preds[j][anchor_status[j]])




if __name__ == '__main__':
    # Load data
    images, class_labels, bboxes, image_scales = dp.load_sfrc_data('/data/')

    # Define model
    base_channels = (1,16,32)
    base_depths = (1,1,1)
    scale_channels = (1,)
    scale_depths = (1,)
    scales = (0.75, 0.5, 0.25, 0.1)
    ratios = (1.0, 2.0, 0.5)
    num_classes = 3
    network = ssm.full_SSD(base_channels, base_depths, scale_channels, scale_depths, scales, ratios, num_classes)

    loc_criterion = nn.SmoothL1Loss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=0.001, params=network.parameters())

    # Train model
    train_ssmd(network, loc_criterion, cls_criterion, optimizer, images, bboxes, 1, 16)
