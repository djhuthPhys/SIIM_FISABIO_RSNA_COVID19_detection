"""
A Single-shot Multibox Detection model for the SIIM FISABIO RSNA COVID 19 detection competition

Author: Dawson Huth
06-17-2021
"""

import torch
import torchvision

from torch import nn
from collections import OrderedDict
from torch.nn import functional as F
from d2l import torch as d2l


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

# Downsampling block
def down_sample_block(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1)))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

# Base Network
def base_net():
    blks = []
    num_filters = [1, 8, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blks.append(down_sample_block(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blks)

def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_block(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_block(128, 128)
    return blk

# Forward prop function for blocks
def blk_forward(X, blk, size, ratio, class_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    class_preds = class_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, class_preds, bbox_preds)

# Some global variables
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

# Copied SSD model from 'Dive into Deep Learning
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(
                self, f'cls_{i}',
                class_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}',
                    bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, class_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], class_preds[i], bbox_preds[i]= blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        class_preds = concatenate_preds(class_preds)
        class_preds = class_preds.reshape(class_preds.shape[0], -1,
                                          self.num_classes + 1)
        bbox_preds = concatenate_preds(bbox_preds)
        return anchors, class_preds, bbox_preds
