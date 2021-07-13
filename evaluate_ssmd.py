import torch

import data_processing as dp
import sfrc19_ssmd_model as ssm





if __name__ == '__main__':
    # Load SSMD model
    base_channels = (1,32,32,32)
    base_depths = (1,1,1,1)
    scale_channels = (32,16,8)
    scale_depths = (1,1,1)
    scales = (0.75, 0.5, 0.25, 0.1)
    ratios = (1.0, 2.0, 0.5)
    num_classes = 1

    model = ssm.full_SSD(base_channels, base_depths, scale_channels,
                         scale_depths, scales, ratios, num_classes)
    model.load_state_dict(torch.load('./sfrc_detector_model.pth'))
    model.eval()

    # Load data
    images = dp.load_