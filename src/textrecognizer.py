# Marakulin Andrey https://github.com/Annndruha
# 2023

import os
import time
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_full_text(response):
    texts = response.text_annotations
    s = texts[0].description
    s = s.replace('à', 'á')
    s = s.replace('ä', 'ā')
    s = s.replace('ë', 'é')
    s = s.replace('ü', 'ū')
    s = s.replace('e', 'ə')
    s = s.replace('\n', ' ')
    return s


# print(get_full_text(response))  # Текст с минимумом эвристик


def get_sub_points(a, l, top_padding=4, bottom_padding=4, right_padding=3, left_padding=-1):
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


def matcher(img, letters, threshold=0.7):
    for letter in letters:
        try:
            template = cv2.imread(f'data/{letter}.jpg')
            if template is None:
                raise Exception('Cant read')
            if np.max(cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)) > threshold:
                return letter
        except Exception as err:
            print('Exception', err, letter)


def replacer(img, letter):
    img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, None, value=(255, 255, 255))
    rr = None
    letter = letter.lower()
    letter = letter.replace('à', 'á')
    letter = letter.replace('ä', 'ā')
    letter = letter.replace('ě', 'š')
    letter = letter.replace('j', 'ǰ')
    letter = letter.replace('e', 'ə')
    letter = letter.replace('ü', 'ū')
    letter = letter.replace('J', 'ǰ')
    letter = letter.replace('j', 'ǰ')

    # if letter == 'a':
    #     rr = matcher(img, ['ā́'], 0.8)
    #     if rr is None:
    #         rr = matcher(img, ['ā'], 0.6)
    #         if rr is None:
    #             rr = matcher(img, ['á'], 0.9)
    # elif letter in ['ā', 'á']:
    #     rr = matcher(img, ['ā́', 'ā', 'á'], 0.7)
    if letter in ['ā́', 'á', 'a']:
        rr = matcher(img, ['ā́', 'ā', 'á'], 0.8)
        if rr is None and letter == 'a':
            rr = matcher(img, ['ā'], 0.62)
    elif letter in ['ə́', 'é', 'e']:
        rr = matcher(img, ['ə́', 'é', 'ə'])
    elif letter in ['ṣ̌', 's']:
        rr = matcher(img, ['ṣ̌', 's'], 0.8)
    elif letter in ['k']:
        rr = matcher(img, ['ḱ'], 0.88)
    elif letter in ['b', '6', 'o']:
        rr = matcher(img, ['ó'], 0.80)
    elif letter in ['ū́', 'ū', 'ú', 'u']:
        rr = matcher(img, ['ū́', 'ū', 'ú'], 0.75)
    if rr is not None:
        letter = rr
    # cv2.imwrite(f'texts/{letter}_{time.time()}.jpg', img)
    return letter


def ocr(image_path, response):
    img = cv2.imread(image_path)
    texts = response.text_annotations
    result_text = []
    for text in texts[1:]:
        pts = np.array([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        polys = get_sub_points(pts, len(text.description), right_padding=0)
        word_letters = []
        for i, (letter, poly) in enumerate(zip(text.description, polys)):
            crop_img = img[poly[0][1]:poly[2][1], poly[0][0]:poly[2][0]]
            letter = replacer(crop_img, letter)
            word_letters.append(letter)
        result_text.append(''.join(word_letters))

    text = ''
    for i, word in enumerate(result_text):
        if word in ['.', ',', '[', ']', ":", '-', '"', '?', '!']:
            if result_text[i - 1] in [':']:
                text += ' ' + word
            else:
                text += word
        else:
            if result_text[i - 1] in ['-', '"']:
                text += word
            else:
                text += ' ' + word
    return text


if __name__ == '__main__':
    IMG_PATH = "tests/text1/img.png"
    RESP_PATH = "tests/text1/google_response.pickle"

    with open(RESP_PATH, "rb") as f:
        from google.cloud import vision  # This 'unused' import used for pickle.load

        response = pickle.load(f)
    print(ocr(IMG_PATH, response))
