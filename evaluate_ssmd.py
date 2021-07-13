import torch

import data_processing as dp
import sfrc19_ssmd_model as ssm

# Train on GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('\nTraining on GPU')
else:
    device = torch.device('cpu')
    print('\nTraining on CPU')


if __name__ == '__main__':
    # Load SSMD model
    base_channels = (1,32,32,32)
    base_depths = (1,1,1,1)
    scale_channels = (32,16,8)
    scale_depths = (1,1,1)
    scales = (0.75, 0.5, 0.25, 0.1)
    ratios = (1.0, 2.0, 0.5)
    num_classes = 1
    batch_size = 64

    model = ssm.full_SSD(base_channels, base_depths, scale_channels,
                         scale_depths, scales, ratios, num_classes)
    model.load_state_dict(torch.load('./sfrc_detector_model.pth'))
    model.eval()

    # Load data
    images, study_ids, image_ids = dp.load_test_images('./data/rescaled_test')

    # Generate predictions
    for i in range(0, images.size()[0], batch_size):
        # Define batch
        if (i + batch_size) > images.size()[0]:
            x_batch = images[i:images.size()[0], :, :, :].to(device)
        else:
            x_batch = images[i:i + batch_size, :, :, :].to(device)

        conf_scores, bbox_preds = model(x_batch)

    # Non-maximal suppression of confidence scores
