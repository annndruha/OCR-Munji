# Marakulin Andrey https://github.com/Annndruha
# 2023

import os
import time
import pickle
import shutil
from google.cloud.vision import AnnotateImageResponse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from mapping import mapping, get_sub_polys


def postprocess_text(img_path: str, response: AnnotateImageResponse, j: int):
    img = cv2.imread(img_path)
    texts = response.text_annotations
    result_text = []
    for it, text in enumerate(texts[1:]):
        pts = np.array([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        polys = get_sub_polys(pts, len(text.description), right_padding=0)
        word_letters = []
        for i, (letter, poly) in enumerate(zip(text.description, polys)):
            letter = mapping(img, poly, letter, j)
            j += 1
            word_letters.append(letter)
        result_text.append(''.join(word_letters))

        if it + 1 < len(texts[1:]):
            next_poly = np.array([(vertex.x, vertex.y) for vertex in texts[it+2].bounding_poly.vertices])
            if pts[1][0] - next_poly[1][0] > 500:
                result_text.append('\n')
    text = ''
    for i, word in enumerate(result_text):
        if word == '\n':
            text += word
        elif word in ['.', ',', '[', ']', ":", '-', '"', '?', '!']:
            if result_text[i - 1] in [':']:
                text += ' ' + word
            else:
                text += word
        else:
            if result_text[i - 1] in ['-', '"', '\n']:
                text += word
            else:
                text += ' ' + word
    return text


# def clarifying

    # shutil.rmtree('letters', ignore_errors=True)
if __name__ == '__main__':
    # IMG_PATH = "tests/text1/img.png"
    # RESP_PATH = "tests/text1/google_response.pickle"
    # with open(RESP_PATH, "rb") as f:
    #     response: AnnotateImageResponse = pickle.load(f)
    # text = postprocess_text(IMG_PATH, response, 100000)
    # with open(IMG_PATH.replace('img.png', 'result.txt'), 'w') as f:
    #     f.write(text)
    #
    #
    # IMG_PATH = "tests/text2/img.png"
    # RESP_PATH = "tests/text2/google_response.pickle"
    # with open(RESP_PATH, "rb") as f:
    #     response: AnnotateImageResponse = pickle.load(f)
    # text = postprocess_text(IMG_PATH, response, 200000)
    # with open(IMG_PATH.replace('img.png', 'result.txt'), 'w') as f:
    #     f.write(text)

    IMG_PATH = "tests/pages_20-21/img.png"
    RESP_PATH = "tests/pages_20-21//google_response.pickle"
    with open(RESP_PATH, "rb") as f:
        response: AnnotateImageResponse = pickle.load(f)
    text = postprocess_text(IMG_PATH, response, 300000)
    with open(IMG_PATH.replace('img.png', 'result.txt'), 'w') as f:
        f.write(text)

    img = cv2.imread(IMG_PATH)
    texts = response.text_annotations
    for text in texts[1:]:
        pts = np.array([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        polys = get_sub_polys(pts, len(text.description))
        for poly in polys:
            img = cv2.polylines(img, [poly], True, (0, 0, 255), 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
