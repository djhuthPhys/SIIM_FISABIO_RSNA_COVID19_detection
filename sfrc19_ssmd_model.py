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


# Scaled feature extraction network
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


    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x


class scaling_network(nn.Module):
    """
    Defines the network responsible for the scaled feature mapping
    """
    def __init__(self, block_channels, block_depths, block=scaling_block, *args, **kwargs):
        """

        :param block_channels: tuple of number of channels used in each block
        :param block_depths: tuple of the number of layers used in each block
        :param block: defines which block to use for the network
        :param args:
        :param kwargs:
        """
        super(scaling_network, self).__init__()

        self.block_channels, self.block_depths, self.block = block_channels, block_depths, block

        self.pooling = nn.MaxPool2d(2)

        self.in_out_pairs = list(zip(block_channels, block_channels[1:]))

        self.blocks = nn.ModuleList([
            *[block(in_channels, out_channels, block_depth=block_depth, *args, **kwargs)
              for (in_channels, out_channels), block_depth in zip(self.in_out_pairs, block_depths[1:])]
        ])

    def forward(self, x):
        block_outputs = []
        for block in self.blocks:
            x = block(x)
            x = self.pooling(x)
            block_outputs.append(x)
        return block_outputs


def scaling_model(block_channels, block_depths):
    return scaling_network(block_channels, block_depths)


# Base and scaling combined
class ssd_convs(nn.Module):
    """
    Combines the base and scaling models into a single model and returns the base output and scaling outputs as a list
    of torch tensors in order of their evaluation.
    """
    def __init__(self, base_channels, base_depths, scale_channels, scale_depths):
        """

        :param base_channels: tuple of number of channels output in each block of the base model
        :param base_depths: tuple of the number of layers used in each block of the base model
        :param scale_channels: tuple of number of channels output in each block of the scaling model
        :param scale_depths: tuple of the number of layers used in each block of the scaling model
        """
        super(ssd_convs, self).__init__()

        self.base_channels, self.base_depths = base_channels, base_depths

        self.scale_depths = (1,) + scale_depths

        self.scale_channels = (base_channels[-1],) + scale_channels

        assert base_channels[-1] == self.scale_channels[0]

        self.base_network = base_network(self.base_channels, self.base_depths)

        self.scaling_network = scaling_network(self.scale_channels, self.scale_depths)

    def forward(self, x):
        base_out = self.base_network(x)
        scale_out = self.scaling_network(base_out)
        return [base_out, scale_out]


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


# Generate anchors
def generate_anchors(data_list, scales, ratios):
    """
    Uses the SSD network output to generate anchor boxes for object detection
    :param data_list: list of output tensors from SSD network, base output is element 0; scaled output is list in
                      element 1
    :param scales: list of scales to use in each tensor output, scale of output sizes normalized to 1
    :param ratios: list of aspect ratios to use in each tensor output
    :return: anchors: list of anchor boxes, each element corresponds to each output of network
    """
    anchors = [d2l.multibox_prior(data_list[0], sizes=scales, ratios=ratios)]
    # Reshape to (features height, features width, # anchors centered on same pixel, 4)
    height = data_list[0].size()[2]
    width = data_list[0].size()[3]
    num_anchors = len(scales) + len(ratios) - 1

    anchors = [anchors[0].reshape(height, width, num_anchors, 4)]
    anchors[0][:, :, :, 2:4] = anchors[0][:, :, :, 2:4] - anchors[0][:, :, :, 0:2]  # Change to height and width format
    anchors[0] = torch.transpose(anchors[0], 0, 2)  # Get in shape (# anchors per pixel, 4, image_size, image_size)
    anchors[0] = torch.transpose(anchors[0], 1, 3)


    for i in range(len(data_list[1])):
        anchors_tmp = d2l.multibox_prior(data_list[1][i], sizes=scales, ratios=ratios)
        # Reshape
        height = data_list[1][i].size()[2]
        width = data_list[1][i].size()[3]
        anchors_tmp = anchors_tmp.reshape(height, width, num_anchors, 4)
        anchors_tmp[:, :, :, 2:4] = anchors_tmp[:, :, :, 2:4] - anchors_tmp[:, :, :, 0:2]  # Change to height and width format
        anchors_tmp = torch.transpose(anchors_tmp, 0, 2) # Get in shape (# anchors per pixel, 4, image_size, image_size)
        anchors_tmp = torch.transpose(anchors_tmp, 1, 3)
        anchors.append(anchors_tmp)

    return anchors


# Full network
class full_SSD(nn.Module):
    """
    Combines the base model, scaling model, anchor generation, and detection heads into a single model that can be
    trained
    """
    def __init__(self, base_channels, base_depths, scale_channels, scale_depths, scales, ratios, num_classes):
        """

        :param base_channels: tuple of number of channels output in each block of the base model
        :param base_depths: tuple of the number of layers used in each block of the base model
        :param scale_channels: tuple of number of channels output in each block of the scaling model
        :param scale_depths: tuple of the number of layers used in each block of the scaling model
        :param scales: list of scales to use in each tensor output, scale of output sizes normalized to 1
        :param ratios: list of aspect ratios to use in each tensor output
        """
        super(full_SSD, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(scales) + len(ratios) - 1
        self.scales, self.ratios = scales, ratios

        self.features = ssd_convs(base_channels, base_depths, scale_channels, scale_depths)

        self.output_channels = (base_channels[-1],) + scale_channels

        self.detectors = nn.ModuleList([detection_head(self.output_channels[i], self.num_anchors, self.num_classes)
                                        for i in range(len(self.output_channels))])


    def forward(self, x):
        # Pass through base model and generate anchor boxes
        x = self.features(x)
        anchors = generate_anchors(x, self.scales, self.ratios)

        # Loop through detector heads to get class and bbox predictions
        base_classes, base_bboxes = self.detectors[0](x[0])
        conf_scores, bbox_preds = [base_classes], [base_bboxes]
        for i in range(len(x[1])):
            classes_tmp, bboxes_tmp = self.detectors[i+1](x[1][i])
            conf_scores.append(classes_tmp)
            bbox_preds.append(bboxes_tmp)

        return conf_scores, bbox_preds, anchors
