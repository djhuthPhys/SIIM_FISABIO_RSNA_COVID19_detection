"""
Training script for the SFRC19 detection SSD model
"""

import data_processing as dp

if __name__ == '__main__':
    # Load data
    images, class_labels, bboxes, scales = dp.load_images('./data')