"""
Training script for the SFRC19 detection SSD model
"""
import torch
import torch.nn as nn

import data_processing as dp
import sfrc19_ssmd_model as ssm

# Train on GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('\nTraining on GPU')
else:
    device = torch.device('cpu')
    print('\nTraining on CPU')



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
    -- has size (batch_size, # anchors per pixel, 4, feature_size, feature_size)
    :return: iou: torch tensor with IoU values
             offsets: torch tensor with bbox prediction offsets
             real_bboxes: tensor of bboxes that actually show up in the images
    """

    # Get labeled bboxes from bbox_labels into tensor of shape (# bboxes, 4 [x, y, width, height])
    bool_tensor = (torch.tensor([0]).to(device) != torch.sum(bbox_labels, dim=2))
    real_bboxes = bbox_labels[0][bool_tensor[0]]
    for i in range(bbox_labels.size()[0]-1):
        real_bboxes = torch.cat((real_bboxes, bbox_labels[i+1][bool_tensor[i+1]]))

    # Get shape of bboxes compatible with adjusted anchors
    anchors_orig = torch.transpose(anchors_orig,0,1).unsqueeze(0)
    real_bboxes = real_bboxes.unsqueeze(2).unsqueeze(2).unsqueeze(2)

    # # Define box and anchor parameters
    x_left = torch.max(anchors_orig[:,0,:,:,:], real_bboxes[:,1,:,:,:])
    y_top = torch.max(anchors_orig[:,1,:,:,:], real_bboxes[:,2,:,:,:])
    x_right = torch.min(anchors_orig[:,0,:,:,:] + anchors_orig[:,2,:,:,:],
                        real_bboxes[:,1,:,:,:] + real_bboxes[:,3,:,:,:])
    y_bottom = torch.min(anchors_orig[:,1,:,:,:] + anchors_orig[:,3,:,:,:],
                         real_bboxes[:,2,:,:,:] + real_bboxes[:,4,:,:,:])

    # Check if boxes overlap
    bool_tensor = torch.logical_or(torch.le(x_right, x_left), torch.le(y_bottom, y_top))

    # Calculate intersection
    intersection = (x_right - x_left) * (y_bottom - y_top)
    intersection[bool_tensor] = 0.0  # If boxes don't overlap intersection is 0

    # Calculate Union
    anchor_area = anchors_orig[:,2,:,:,:] * anchors_orig[:,3,:,:,:]
    box_area = real_bboxes[:,3,:,:,:] * real_bboxes[:,4,:,:,:]
    union = (anchor_area + box_area) - intersection

    # Calculate IoU
    iou = intersection/union

    # Calculate anchor box offsets
    x_min_off = (anchors_orig[:,0,:,:,:] - real_bboxes[:,1,:,:,:]).unsqueeze(dim=1)
    y_min_off = (anchors_orig[:,1,:,:,:] - real_bboxes[:,2,:,:,:]).unsqueeze(dim=1)
    width_off = (anchors_orig[:,2,:,:,:] - real_bboxes[:,3,:,:,:]).unsqueeze(dim=1)
    height_off = (anchors_orig[:,3,:,:,:] - real_bboxes[:,4,:,:,:]).unsqueeze(dim=1)
    offsets = torch.cat((x_min_off, y_min_off, width_off, height_off), dim=1)

    return iou, offsets, real_bboxes


def get_positive_anchors(bbox_labels, anchors_orig):
    """
    Labels the anchor boxes with IoUs over a threshold of 0.5 as positive (1) and under a threshold of 0.5 as
    negative (0), returns the anchor box offsets needed to match bounding box labels
    :param bbox_labels: List of ground truth bounding box tensors
    :param anchors_orig: list of default anchor box tensors
    :return: anchor_status, anchor_offsets, real_bbox_list, iou_list
    """
    threshold = 0.5
    iou_list = []
    anchor_status = []
    anchor_offsets = []
    real_bbox_list = []
    for i in range(len(anchors_orig)):
        iou, offsets, real_bboxes = process_anchors(bbox_labels, anchors_orig[i])
        iou_list.append(iou)

        # Format iou to match bbox_labels and adjusted_anchors
        anchor_status.append(iou > threshold)
        real_bbox_list.append(real_bboxes)
        anchor_offsets.append(offsets)

    return anchor_status, anchor_offsets


def process_confs(conf_scores, anchors, bbox_labels, batch_size, max_boxes):
    """
    Processes the class scores to
    :param conf_scores: List of confidence scores for classes
    :param anchors: List of anchor values
    :param bbox_labels: bbox labels in batch
    :param batch_size: Batch size used in training
    :param max_boxes: The maximum number of boxes that appear in the data set for a single image
    :return: processed_scores
    """

    anchors_per_pixel = int(conf_scores[0].size()[1]/2)
    shaped_scores = []
    cls_labels = []
    for i in range(len(conf_scores)):
        feature_size = conf_scores[i].size()[-1]
        # Reshape confidence scores to
        # (batch_size, num_classes + 1, # anchors per pixel, feature_size, feature_size)
        scores_tmp = conf_scores[i].reshape(batch_size, 2, anchors_per_pixel, feature_size,
                                            feature_size)
        shaped_scores.append(scores_tmp)

        # Get IoU threshold values in the shape (batch_size, max_boxes, num_classes + 1, # anchors per pixel, feature size, feature_size)
        _, status = exhaustive_iou(anchors[i], bbox_labels, max_boxes)

        # Generate class labels list based on IoU score
        cls_labels.append(torch.sum(status, dim=1).long())

    return shaped_scores, cls_labels


def exhaustive_iou(anchors, bbox_labels, max_boxes):
    """
    Calculates the IoU scores in the shape (batch_size, max_boxes, num_classes + 1, # anchors per pixel, feature size,
    feature_size) for the confidence score labeling
    :param anchors:
    :param bbox_labels:
    :param max_boxes
    :return:
    """

    # Get shape of bboxes compatible with adjusted anchors
    anchors = torch.transpose(anchors,0,1).unsqueeze(0).unsqueeze(0).repeat(1,max_boxes,1,1,1,1)
    bbox_labels = bbox_labels.unsqueeze(3).unsqueeze(3).unsqueeze(3)

    # Calculate IoU
    # Define box and anchor parameters
    x_left = torch.max(anchors[:,:,0,:,:,:], bbox_labels[:,:,1,:,:,:])
    y_top = torch.max(anchors[:,:,1,:,:,:], bbox_labels[:,:,2,:,:,:])
    x_right = torch.min(anchors[:,:,0,:,:,:] + anchors[:,:,2,:,:,:],
                        bbox_labels[:,:,1,:,:,:] + bbox_labels[:,:,3,:,:,:])
    y_bottom = torch.min(anchors[:,:,1,:,:,:] + anchors[:,:,3,:,:,:],
                         bbox_labels[:,:,2,:,:,:] + bbox_labels[:,:,4,:,:,:])

    # Check if boxes overlap
    bool_tensor = torch.logical_or(torch.le(x_right, x_left), torch.le(y_bottom, y_top))

    # Calculate intersection
    intersection = (x_right - x_left) * (y_bottom - y_top)
    intersection[bool_tensor] = 0.0  # If boxes don't overlap intersection is 0

    # Calculate Union
    anchor_area = anchors[:,:,2,:,:,:] * anchors[:,:,3,:,:,:]
    box_area = bbox_labels[:,:,3,:,:,:] * bbox_labels[:,:,4,:,:,:]
    union = (anchor_area + box_area) - intersection

    # Calculate IoU
    iou = intersection/union
    status = torch.gt(iou, 0.5)

    return iou, status


def train_ssmd(model, bbox_criterion, class_criterion, optimization, X, Y, epochs, batch_size):
    """
    Training function for the SSMD model to predict opacity and bounding boxes
    :param model: SSMD model to train
    :param bbox_criterion: The loss function used for bounding box prediction
    :param class_criterion: The loss function used for opacity prediction
    :param optimization: Optimizing algorithm to use
    :param X: Input images for training
    :param Y: Bounding box and opacity labels
    :param epochs: Number of epochs to train for
    :param batch_size: Batch size used in training
    :return: predictions
    """
    max_boxes = Y.size()[1]
    running_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        permutation = torch.randperm(X.size()[0])

        for i in range(0, X.size()[0], batch_size)[0:-1]:
            # Zero out gradients
            optimization.zero_grad()

            # Define batch
            indices = permutation[i:i+batch_size]
            x_batch, y_batch = X[indices, :, :, :].to(device), Y[indices, :, :].to(device)

            # Pass data through network
            conf_scores, bbox_preds, anchors = model(x_batch)

            # Reshape bbox_preds for localization loss calculation
            for k in range(len(bbox_preds)):
                anchors_per_pixel = int(bbox_preds[k].size()[1]/4)
                feature_size = bbox_preds[k].size()[-1]
                bbox_preds[k] = bbox_preds[k].reshape((batch_size, 4, anchors_per_pixel, feature_size,
                                                       feature_size)).unsqueeze(1).repeat(1,max_boxes,1,1,1,1)

            # Determine positive and negative boxes/anchors and calculate anchor offsets
            # Get positive offsets and bbox predictions for loss calculation
            anchor_status, anchor_offsets = get_positive_anchors(y_batch, anchors)
            for j in range(len(anchors)):
                anchor_offsets[j] = anchor_offsets[j][anchor_status[j].unsqueeze(1).repeat(1,4,1,1,1)]
                _, bbox_status = exhaustive_iou(anchors[j], y_batch, max_boxes)
                bbox_preds[j] = bbox_preds[j][bbox_status.unsqueeze(2).repeat(1,1,4,1,1,1)]\

            # Process confidence scores to calculate losses
            conf_scores, box_labels = process_confs(conf_scores, anchors, y_batch, batch_size, max_boxes)

            # Calculate losses and update parameters
            loc_loss = 0.0
            cls_loss = 0.0
            for m in range(len(anchors)):
                loc_loss += bbox_criterion(bbox_preds[m], anchor_offsets[m])
                cls_loss += class_criterion(conf_scores[m], box_labels[m])
            loss = loc_loss + cls_loss
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1
            if (i/batch_size) % 20 == 19:
                print('Epoch %i, batch %i: training loss = %.8f' % (epoch, i/batch_size + 1, running_loss/num_batches))


if __name__ == '__main__':
    # Load data
    images, class_labels, bboxes, image_scales = dp.load_sfrc_data('./data')

    # Define model
    base_channels = (1,64,64,64)
    base_depths = (1,1,1)
    scale_channels = (1,1,1)
    scale_depths = (64,64,64)
    scales = (0.75, 0.5, 0.25, 0.1)
    ratios = (1.0, 2.0, 0.5)
    num_classes = 1
    network = ssm.full_SSD(base_channels, base_depths, scale_channels,
                           scale_depths, scales, ratios, num_classes).to(device)

    loc_criterion = nn.SmoothL1Loss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=0.001, params=network.parameters())

    # Train model
    batch_size = 32
    num_epochs = 3
    train_ssmd(network, loc_criterion, cls_criterion, optimizer, images, bboxes, num_epochs, batch_size)
