# Marakulin Andrey https://github.com/Annndruha
# 2023

import cv2
import numpy as np
from google.cloud.vision import AnnotateImageResponse

from mapping import mapping, get_sub_polys


def process_text(img_path: str, response: AnnotateImageResponse):
    img = cv2.imread(img_path)
    texts = response.text_annotations
    result_text = []
    for it, text in enumerate(texts[1:]):
        pts = np.array([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        polys = get_sub_polys(pts, len(text.description), right_padding=0)
        word_letters = []
        for i, (letter, poly) in enumerate(zip(text.description, polys)):
            letter = mapping(img, poly, letter)
            word_letters.append(letter)
        result_text.append(''.join(word_letters))

        if it + 1 < len(texts[1:]):
            next_poly = np.array([(vertex.x, vertex.y) for vertex in texts[it + 2].bounding_poly.vertices])
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
