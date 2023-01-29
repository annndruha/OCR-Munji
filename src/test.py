# Marakulin Andrey https://github.com/Annndruha
# 2023

import os
import time
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt

from processing_rules import mapping, get_sub_polys


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


def create_dataset(img_path, response):
    img = cv2.imread(img_path)
    texts = response.text_annotations
    for text in texts[1:]:
        pts = np.array([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        polys = get_sub_polys(pts, len(text.description), right_padding=0)
        for i, (letter, poly) in enumerate(zip(text.description, polys)):
            letter = mapping(img, poly, letter)

            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
            crop_img = img[y0 - 15:y2 + 15, x0:x2]
            save_letter(letter, crop_img, 'all_letters')


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

    # import cv2
    # import numpy as np
    # img = cv2.imread(IMG_PATH)
    # texts = response.text_annotations
    # for text in texts[1:]:
    #     pts = np.array([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
    #     polys = get_sub_points(pts, len(text.description))
    #     for poly in polys:
    #         img = cv2.polylines(img, [poly], True, (0, 0, 255), 1)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()