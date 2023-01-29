# Marakulin Andrey https://github.com/Annndruha
# 2023

import os
import time
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import save_letter


def get_sub_polys(a, l, top_padding=0, bottom_padding=0, right_padding=0, left_padding=0):
    x_top = np.linspace(a[0][0], a[1][0], l + 1)
    x_bottom = np.linspace(a[3][0], a[2][0], l + 1)
    y_top = np.linspace(a[0][1], a[1][1], l + 1)
    y_bottom = np.linspace(a[3][1], a[2][1], l + 1)
    polys = []
    for i in range(l):
        pts = [[x_top[i] - left_padding, y_top[i] - top_padding],
               [x_top[i + 1] + right_padding, y_top[i + 1] - top_padding],
               [x_bottom[i + 1] + right_padding, y_bottom[i + 1] + bottom_padding],
               [x_bottom[i] - left_padding, y_bottom[i] + bottom_padding]]
        polys.append(np.array(pts, dtype=np.int32))
    return polys


def mapping(img, poly, letter: str) -> str:
    try:
        d = {
            'S': 'š' + u'\u0323',
            'j': 'ǰ',
            'J': 'ǰ',
            'e': 'ə',  # Wrong, but statistic
            'é': 'ə́',  # Wrong, but statistic
            'ś': 'ə' + u'\u0301',

            'š': __under_dot,
            'ž': __under_dot,
            'č': __under_dot,
            'u': __upper_comb_u,
            'ú': __upper_comb_u,
            'ū': __upper_comb_u,
            'ü': __upper_comb_u,
        }
        res = d[letter]
        if callable(res):
            return res(img, poly, letter)  # Process special letter
        else:
            return d[letter]  # Mistake case
    except KeyError:
        return letter  # Work well or unknown


def __under_dot(img, poly, letter):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    crop_img = img[y2 - 2:y2 + 18, x0:x2]
    crop_img = cv2.copyMakeBorder(crop_img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, None, value=(230, 255, 255))

    template = cv2.imread('templates/dot.jpg')
    prob = np.max(cv2.matchTemplate(crop_img, template, cv2.TM_CCOEFF_NORMED))
    if prob > 0.45:
        letter = letter + u'\u0323'
    # save_letter(letter, crop_img)
    return letter


def __upper_comb_u(img, poly, letter):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    crop_img = img[y0:y2, x0:x2]
    crop_img = cv2.copyMakeBorder(crop_img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, None, value=(230, 255, 255))

    # template = cv2.imread('templates/dot.jpg')
    # prob = np.max(cv2.matchTemplate(crop_img, template, cv2.TM_CCOEFF_NORMED))
    # if prob > 0.45:
    #     letter = letter + u'\u0323'
    save_letter(letter, crop_img)
    return letter
