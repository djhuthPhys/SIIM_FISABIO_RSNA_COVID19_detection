import glob
import os

import numpy as np
import pandas as pd
import pydicom
import torch
from skimage.transform import resize
from tqdm import tqdm


def load_labels(path):
    """
    Loads the train_image_level.csv and train_study_labels.csv files into a pandas dataframe. Data is then processed
    into pytorch tensors for training a model
    :param path: Path to directory where files are located
    :return: bbox_labels, study_labels, example_ids
    """

    # Read in label data
    image_data= pd.read_csv(path + '/train_image_level.csv')
    study_data = pd.read_csv(path + '/train_study_level.csv')

    # Process IDs to return
    image_data['id'] = image_data['id'].apply(lambda x: x.split('_')[0])
    example_ids = pd.concat([image_data['id'], image_data['StudyInstanceUID']], axis=1)

    # Find maximum number of bounding boxes in a single image
    max_boxes = max([len(image_data.label[i].split())//6 for i in range(image_data.shape[0])])

    # Process image level label data into pytorch tensor format for training
    # bbox_labels are in format (# images, maximum # boxes, # coordinates [xmin, ymin, xmax, ymax])
    bbox_labels = torch.zeros(image_data.shape[0], max_boxes, image_data.shape[1]+1)
    print('Loading bounding boxes')
    for image in tqdm(range(len(image_data.label))):
        labels_list = [float(label) for label in image_data.label[image].split() if label.replace('.', '').isnumeric()]
        for box in range(len(labels_list)//5):
            if 'none' in image_data.label[image].split():
                opacity = 0.0
            else:
                opacity = 1.0
            x_min = labels_list[5 * box + 1]
            y_min = labels_list[5 * box + 2]
            width = labels_list[5 * box + 3] - labels_list[5 * box + 1]
            height = labels_list[5 * box + 4] - labels_list[5 * box + 2]
            bbox_coords = torch.tensor([opacity, x_min, y_min, width, height])
            bbox_labels[image, box, :] = bbox_coords

    # Process study level label data into pytorch tensor format for training
    print('Processing study labels')
    study_labels = torch.zeros(study_data.shape[0],study_data.shape[1] - 1)
    for study in tqdm(range(len(study_data.id))):
        study_labels[study, 0] = study_data['Negative for Pneumonia'][study]
        study_labels[study, 1] = study_data['Typical Appearance'][study]
        study_labels[study, 2] = study_data['Indeterminate Appearance'][study]
        study_labels[study, 3] = study_data['Atypical Appearance'][study]

    return bbox_labels, study_labels, example_ids


def downsample_images(path, size=512):
    """
    Processes all images into a size x size array for training. X and Y scale factors from resizing are saved in
    IMAGE_NAME_scale.npy in the same directory as the image IMAGE_NAME.npy
    :param path: Path to original images
    :param size: size of image to make
    :return: None
    """

    _, _, example_ids = load_labels('./data')

    for image_num in range(len(example_ids)):

        study_id = example_ids['StudyInstanceUID'][image_num]
        image_id = example_ids['id'][image_num]

        image_path = glob.glob(path + '/' + study_id + '/*/' + image_id + '.dcm')[0]
        image_data = pydicom.dcmread(image_path).pixel_array

        # Resize image to size x size array and get scaling information
        resized_image = resize(image_data, (size, size), anti_aliasing=True, preserve_range=True)
        y_scale = image_data.shape[0]/size
        x_scale = image_data.shape[1]/size
        scaling = np.array([y_scale, x_scale])

        # Get data set type
        if 'train' in image_path:
            set_type = 'train'
        elif 'test' in image_path:
            set_type = 'test'
        else:
            print('\nUnexpected data set type')
            set_type = 'bad_type'

        # Path formatting and directory creation
        sub_directory_names = image_path.split('./data/' + set_type)[1].split('\\')[0:2]
        sub_directory = sub_directory_names[0] + '/' + sub_directory_names[1]
        if not os.path.exists('./data/rescaled_' + set_type + sub_directory):
            os.makedirs('./data/rescaled_' + set_type + sub_directory)
        rescale_path = './data/rescaled_' + set_type + '/' + image_path.split('./data/' + set_type)[1].split('.dcm')[0]

        # Save image and scaling to numpy file
        np.save(rescale_path + '.npy', resized_image)
        np.save(rescale_path + '_scale.npy', scaling)
        print('Saving rescaled image ' + image_id + ' from study ' + study_id)

    print('Done!')

    return None


def load_images(path, example_ids):
    """
    Loads in the .dcm images
    :param path: Path to image files
    :param example_ids: The IDs associated with the loaded example data
    :return: image_tensor: Pytorch tensor with image level data in the same order as loaded labels
    """

    # Get image parameters
    num_images = len(example_ids)
    test_image = np.load(glob.glob(path + '/rescaled_train/' + example_ids['StudyInstanceUID'][0] + '/*/*')[0])
    image_size = np.shape(test_image)[0]
    image_tensor = torch.zeros((num_images, 1, image_size, image_size))
    print('\nThere are ' + str(num_images) + ' images in this data set of size '
          + str(image_size) + 'X' + str(image_size) + ' pixels')

    it = 0
    for image_num in tqdm(range(len(example_ids))):

        study_id = example_ids['StudyInstanceUID'][image_num]
        image_id = example_ids['id'][image_num]
        image_path = glob.glob(path + '/rescaled_train/' + study_id + '/*/' + image_id + '.npy')
        image = torch.from_numpy(np.load(image_path[0]))
        image = torch.reshape(image, (1, 1, image_size, image_size))
        image_tensor[image_num,:,:,:] = image
        it += 1

    print(str(it) + ' images of size '
          + str(image_tensor.size()[2]) + 'X' + str(image_tensor.size()[3]) + ' pixels loaded')

    return image_tensor


def load_scaling(path, example_ids):
    """
    Loads the scaling information of the images from when they were resized with downsample_images function. The first
    column are y scaling factors and the second are x scaling factors.
    :param path: Path to scaling files
    :param example_ids: The IDs associated with the loaded example data
    :return:
    """

    num_images = len(example_ids)
    scaling_tensor = torch.zeros((num_images, 2))
    print('\nThere are ' + str(num_images) + ' scaling files in this data set')

    it = 0
    for image_num in tqdm(range(len(example_ids))):

        study_id = example_ids['StudyInstanceUID'][image_num]
        image_id = example_ids['id'][image_num]
        scale_path = glob.glob(path + '/rescaled_train/' + study_id + '/*/' + image_id + '_scale.npy')
        print(study_id)
        print(image_id)
        print(scale_path)
        scales = torch.from_numpy(np.load(scale_path[0]))
        scaling_tensor[image_num,:] = scales
        it += 1

    print(str(it) + ' scaling values loaded')

    return scaling_tensor


def get_class_weights(study_labels):
    """
    Calculates the label weights used in training based on class imbalances. Ratio of population of largest class to
    individual class populations. Largest class will have a weight of 1.
    :param study_labels: Torch tensor containing study level labels
    :return: class_weights: Torch tensor containing class weights
    """

    cumulative_labels = torch.sum(study_labels, dim=0)
    class_weights = torch.max(cumulative_labels)/cumulative_labels

    return class_weights


def load_sfrc_data(path):
    """
    Loads all label, image, and scaling data in consistent order. Data loaded from here is ready for training with SSD
    model
    :param: path: path to data files
    :return: images, class_labels, bboxes
    """

    # Load raw data
    bboxes, class_labels, example_ids = load_labels(path)

    scales = load_scaling(path, example_ids)

    images = load_images(path, example_ids)

    # Process bounding boxes
    print('\nProcessing bounding boxes')
    image_size = images.size()[2]
    exchange_scale = torch.index_select(scales, 1, torch.LongTensor([1,0]))
    bboxes[:, :, 1:5] = bboxes[:, :, 1:5] / exchange_scale.unsqueeze(dim=1).repeat(1,1,2) / image_size  # Normalize bounding box coordinates

    return images, class_labels, bboxes, scales
