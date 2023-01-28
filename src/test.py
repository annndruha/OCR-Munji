# Marakulin Andrey https://github.com/Annndruha
# 2023

import os
import time
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def save_letter(letter, crop_img):
    code = '_'.join([str(ord(i)) for i in letter])
    folder_name = str(code) + f'_{letter}'
    if not os.path.exists(os.path.join('letters', folder_name)):
        os.makedirs(os.path.join('letters', folder_name))
    cv2.imwrite(os.path.join('letters', folder_name, f'{time.time()}.jpg'), crop_img)


def ord353(img, poly):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    crop_img = img[y2 - 2:y2 + 18, x0:x2]
    crop_img = cv2.copyMakeBorder(crop_img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, None, value=(230, 255, 255))

    template = cv2.imread('templates/dot.jpg')
    if np.max(cv2.matchTemplate(crop_img, template, cv2.TM_CCOEFF_NORMED)) > 0.45:
        letter = 'ṣ̌'
    else:
        letter = 'š'
    save_letter(letter, crop_img)
    return letter


def create_dataset(img_path, response):
    img = cv2.imread(img_path)
    texts = response.text_annotations
    result_text = []
    for text in texts[1:]:
        pts = np.array([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        polys = get_sub_points(pts, len(text.description), right_padding=0)
        for i, (letter, poly) in enumerate(zip(text.description, polys)):
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
# ====================================================================================================
            if ord(letter) == 353:
                letter = ord353(img, poly)
# ====================================================================================================
            # crop_img = img[y0-15:y2+15, x0:x2]
            # try:
            #     if not os.path.exists(os.path.join('letters', str(ord(letter)) + f'_{letter}')):
            #         os.makedirs(os.path.join('letters', str(ord(letter)) + f'_{letter}'))
            #     cv2.imwrite(os.path.join('letters', str(ord(letter)) + f'_{letter}', f'{time.time()}.jpg'), crop_img)
            # except OSError:
            #     if not os.path.exists(os.path.join('letters', str(ord(letter)))):
            #         os.makedirs(os.path.join('letters', str(ord(letter))))
            #     cv2.imwrite(os.path.join('letters', str(ord(letter)), f'{time.time()}.jpg'), crop_img)


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
