# Marakulin Andrey https://github.com/Annndruha
# 2023

import os
import time
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_letter(letter, crop_img, folder='letters'):
    folder_code = str('_'.join([str(ord(i)) for i in letter]))
    folder_name = folder_code + f'_{letter}'
    try:
        if not os.path.exists(os.path.join(folder, folder_name)):
            os.makedirs(os.path.join(folder, folder_name))
        cv2.imwrite(os.path.join(folder, folder_name, f'{time.time()}.jpg'), crop_img)
    except OSError:
        if not os.path.exists(os.path.join(folder, folder_code)):
            os.makedirs(os.path.join(folder, folder_code))
        cv2.imwrite(os.path.join(folder, folder_code, f'{time.time()}.jpg'), crop_img)