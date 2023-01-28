# Marakulin Andrey https://github.com/Annndruha
# 2023

import os
import time
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt

# from textrecognizer import get_sub_points, replacer
def get_sub_points(a, l, top_padding=0, bottom_padding=0, right_padding=0, left_padding=0):
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

# print(''.join(sorted(list(set(response.text_annotations[0].description)))))


def create_dataset(img_path, response):
    img = cv2.imread(img_path)
    texts = response.text_annotations
    result_text = []
    for text in texts[1:]:
        pts = np.array([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        polys = get_sub_points(pts, len(text.description), right_padding=0)
        # word_letters = []
        for i, (letter, poly) in enumerate(zip(text.description, polys)):
            crop_img = img[poly[0][1]:poly[2][1], poly[0][0]:poly[2][0]]
            try:
                if not os.path.exists(os.path.join('letters', str(ord(letter)) + f'_{letter}')):
                    os.makedirs(os.path.join('letters', str(ord(letter)) + f'_{letter}'))
                cv2.imwrite(os.path.join('letters', str(ord(letter)) + f'_{letter}', f'{time.time()}.jpg'), crop_img)
            except OSError:
                if not os.path.exists(os.path.join('letters', str(ord(letter)))):
                    os.makedirs(os.path.join('letters', str(ord(letter))))
                cv2.imwrite(os.path.join('letters', str(ord(letter)), f'{time.time()}.jpg'), crop_img)


if __name__ == '__main__':
    IMG_PATH = "tests/text1/img.png"
    RESP_PATH = "tests/text1/google_response.pickle"
    with open(RESP_PATH, "rb") as f:
        from google.cloud import vision  # This 'unused' import used for pickle.load
        response = pickle.load(f)
    create_dataset(IMG_PATH, response)

    IMG_PATH = "tests/text2/img.png"
    RESP_PATH = "tests/text2/google_response.pickle"
    with open(RESP_PATH, "rb") as f:
        from google.cloud import vision  # This 'unused' import used for pickle.load
        response = pickle.load(f)
    create_dataset(IMG_PATH, response)
