"""
These methods below are helpers for preprocessing.
"""

import pandas as pd
import cv2
import random
import numpy as np

def crop_top_and_bottom(image):
    resized = cv2.resize(image[70:140], (64,64),  cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)[:,:,1]


def shift_img(image, random_shift):
    rows, cols = image.shape
    mat = np.float32([[1, 0, random_shift], [0, 1, 0]])
    return cv2.warpAffine(image, mat, (cols, rows))

def load_image(row):
    """
    This is the main workhorse. It takes a dataframe row and loads the image; making
    proper augmentations based on the flags included. Assumes images are in a
    directory called './data'. Also assumes that the dataframe row has gone through
    the proper modifications in the preprocessing step. Refer to the playground
    notebook for more info.
    Args:
        row: dataframe row
    Returns:
        image: nd.array
    """
    image = cv2.imread("./data/{0}".format(row.image.strip()))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_top_and_bottom(image)

    if(row.is_flipped):
        image = cv2.flip(image,1)
    if(row.is_shift):
        image = shift_img(image, row.random_shift)
    return image

def get_processed_dataframes():
    """
    Assumes the proper modifications in the preprocessing step. Refer to the playground
    notebook for more info. Assumes the preprocessed csv driver log is titeled
    'preprocessed_driver_log.csv'
    Returns:
        df: dataframe
    """
    return pd.read_csv('preprocessed_driver_log.csv')
