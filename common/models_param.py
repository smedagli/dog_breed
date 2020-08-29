"""
This module contains common functions and parameters for model training
"""

args_data_augmentation = {'width_shift_range': 0.3,
                          'height_shift_range': 0.3,
                          'horizontal_flip': True,
                          'rescale': True,
                          'rotation_range': 0.5,
                          }  # parameters for data augmentation
