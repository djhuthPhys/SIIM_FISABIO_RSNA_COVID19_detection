"""
A Single-shot Multibox Detection model for the SIIM FISABIO RSNA COVID 19 detection competition

Author: Dawson Huth
06-17-2021
"""

import torch

from torch import nn
from collections import OrderedDict
from d2l import torch as d2l


# Base model classes and base model definition function
class base_layer(nn.Module):
    """
    Defines the basic residual layer of the base SSD network
    """
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        """

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param args:
        :param kwargs:
        :return:
        """
        super(base_layer, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels

        self.conv = nn.Sequential(OrderedDict(
            {
                'conv1': nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1), *args, **kwargs),

                'norm': nn.BatchNorm2d(out_channels),

                'relu': nn.ReLU(),

                'conv2': nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1), *args, **kwargs)
            }
        ))

        self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), *args, **kwargs)

    def forward(self, x):
        residual = x
        if self.apply_mapping:
            residual = self.residual(x)
        x = self.conv(x)
        x += residual
        x = self.relu(self.norm(x))
        return x

    @property
    def apply_mapping(self):
        return self.in_channels != self.out_channels


class base_block(nn.Module):
    """
    Defines the basic block of the base SSD network
    """
    def __init__(self, in_channels, out_channels, layer=base_layer, block_depth=1, *args, **kwargs):
        """

        :param in_channels: number of in channels (only used by first convolution layer in block)
        :param out_channels: number of out channels (used by the rest of the layers)
        :param layer: defines the layer to use when building the block
        :param block_depth: defines the number of layers the block is built with
        :param args:
        :param kwargs:
        """
        super(base_block, self).__init__()

        self.block = nn.Sequential(
            layer(in_channels, out_channels, *args, **kwargs),
            *[layer(out_channels, out_channels, *args, **kwargs)
              for _ in range(block_depth-1)]
        )

    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x


class base_network(nn.Module):
    """
    Defines the base network for the SSD model
    """
    def __init__(self, block_channels, block_depths, block=base_block, *args, **kwargs):
        """

        :param block_channels: tuple of number of channels used in each block
        :param block_depths: tuple of the number of layers used in each block
        :param block: defines which block to use for the network
        :param args:
        :param kwargs:
        """
        super(base_network, self).__init__()

        self.block_channels, self.block_depths, self.block = block_channels, block_depths, block

        self.pooling = nn.MaxPool2d(2)

        self.in_out_pairs = list(zip(block_channels, block_channels[1:]))

        self.blocks = nn.ModuleList([
            block(block_channels[0], block_channels[0], block_depth=block_depths[0], *args, **kwargs),
            *[block(in_channels, out_channels, block_depth=block_depth, *args, **kwargs)
              for (in_channels, out_channels), block_depth in zip(self.in_out_pairs, block_depths[1:])]
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            x = self.pooling(x)
        return x


def base_model(block_channels, block_depths):
    return base_network(block_channels, block_depths)


# Scaled feature extraction
class scaling_layer(nn.Module):
    """
    Defines the layer for extracting features at a new scale from the base network output
    """
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        """

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param args:
        :param kwargs:
        """
        super(scaling_layer, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels

        self.conv = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1), *args, **kwargs),

                'norm': nn.BatchNorm2d(out_channels),

                'relu': nn.ReLU()
            }
        ))

    def forward(self, x):
        return self.conv(x)


class scaling_block(nn.Module):
    """
    Defines the blocks for scaling features. Most blocks will usually only be one layer deep but this is defined for the
    sake of customizability
    """
    def __init__(self, in_channels, out_channels, layer=scaling_layer, block_depth=1, *args, **kwargs):
        """"

        :param in_channels: number of in channels (only used by first convolution layer in block)
        :param out_channels: number of out channels (used by the rest of the layers)
        :param layer: defines the layer to use when building the block
        :param block_depth: defines the number of layers the block is built with
        :param args:
        :param kwargs:
        """
        super(scaling_block, self).__init__()

        self.block = nn.Sequential(
            layer(in_channels, out_channels, *args, **kwargs),
            *[layer(out_channels, out_channels, *args, **kwargs)
              for _ in range(block_depth-1)]
        )

        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        for layer in self.block:
            x = layer(x)
            x = self.pooling(x)
        return x


# Detection head
class detection_head(nn.Module):
    """
    Defines the detection head layers
    """
    def __init__(self, in_channels, num_anchors, num_classes):
        """

        :param in_channels: number of channels output by corresponding feature extraction layer
        :param num_anchors: number of anchor boxes generated per pixel
        :param num_classes: number of classes for the classification problem
        """
        super(detection_head, self).__init__()

        self.in_channels, self.num_anchors, self.num_classes = in_channels, num_anchors, num_classes

        self.class_predictor = nn.Conv2d(in_channels, num_anchors * (num_classes + 1), kernel_size=(3,3), padding=(1,1))

        self.bbox_predictor = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(3,3), padding=(1,1))

    def forward(self, x):
        classes = self.class_predictor(x)
        bboxes = self.bbox_predictor(x)
        return classes, bboxes


# Full forward pass of the SSD network
def SSD_forward(base, scale_depths, X):
    """
    The full forward pass through the SSD network. Outputs class confidence scores and bbox predictions
    :param base: The base model
    :param X: Input data to model
    :return: conf_scores, bbox_preds
    """
    # Anchor box variables
    scales = [0.75, 0.66, 0.5, 0.33, 0.25, 0.1]
    ratios = [1, 2, 0.5, 0.25]
    num_classes = 4
    num_anchors = len(scales) + len(ratios) - 1


    # Pass through base model and generate anchor boxes
    base_output = base(X)
    ####################### Generate anchor boxes
    image_height = base_output.size()[2]
    image_width = base_output.size()[3]
    bbox_scale = torch.tensor((image_width, image_height, image_width, image_height))
    base_anchors = d2l.multibox_prior(base_output, sizes=scales, ratios=ratios)
    scaled_anchors = base_anchors * bbox_scale
    #######################
    base_classes, base_bboxes = detection_head(base_output.size()[1], num_anchors, num_classes)(base_output)

    print(base_output.size())
    print(base_anchors.size())
    print(scaled_anchors.size())
    print(base_classes.size())
    print(base_bboxes.size())

    # Pass outputs from base model through scaled feature extraction


    return None



########################################################################################################################
# Class prediction layer
def class_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=(3,3), padding=(1,1))

# Bounding Box prediction layer
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=(3,3), padding=(1,1))

# Flatten predictions for concatenation
def flatten_preds(pred):
    return torch.flatten(pred.permute(0,2,3,1), start_dim=1)

# Concatenate predictions for loss calculation
def concatenate_preds(preds):
    return torch.cat([flatten_preds(p) for p in preds], dim=1)

# Forward prop function for blocks
def blk_forward(X, blk, size, ratio, class_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    class_preds = class_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, class_preds, bbox_preds)
