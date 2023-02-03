# Marakulin Andrey https://github.com/Annndruha
# 2023

import os
import time

import cv2
import numpy as np

DEBUG = False


# With bounding poly of N-letter word
# make an N polys for every letter
def get_sub_polys(poly, word_length, top_padding=0, bottom_padding=0, right_padding=0, left_padding=0):
    x_top = np.linspace(poly[0][0], poly[1][0], word_length + 1)
    x_bottom = np.linspace(poly[3][0], poly[2][0], word_length + 1)
    y_top = np.linspace(poly[0][1], poly[1][1], word_length + 1)
    y_bottom = np.linspace(poly[3][1], poly[2][1], word_length + 1)
    polys = []
    for i in range(word_length):
        pts = [[x_top[i] - left_padding, y_top[i] - top_padding],
               [x_top[i + 1] + right_padding, y_top[i + 1] - top_padding],
               [x_bottom[i + 1] + right_padding, y_bottom[i + 1] + bottom_padding],
               [x_bottom[i] - left_padding, y_bottom[i] + bottom_padding]]
        polys.append(np.array(pts, dtype=np.int32))
    return polys


# With full img and poly of letter, and letter itself
# make heuristics replacements and correlation comparisons for fix letter
# to Munji correct letter
def mapping(img, poly, letter: str) -> str:
    try:
        letter = __instant_replace(letter)
        d = {
            'e': __check_reversed_e,
            'é': __check_reversed_e_acute,

            'ṣ': __under_dot,
            'š': __under_dot,
            'ž': __under_dot,
            'č': __under_dot,

            'u': __upper_comb_u,
            'ú': __upper_comb_u,
            'ū': __upper_comb_u,
            'ü': __upper_comb_u,

            'a': __upper_comb_u,
            'ä': __upper_comb_u,
            'á': __upper_comb_u,
            'ã': __upper_comb_u,
            'ā': __upper_comb_u,

            'o': __acute_o,
            'ó': __check_reversed_e_acute_o,
            'k': __acute_k,
            'g': __acute_g,
            'i': __acute_i,
        }
        res = d[letter]
        if callable(res):
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
            if y2 - y0 < 10 or x2 - x0 < 10:
                return ''
            return res(img, poly, letter)  # Process special letter
        else:
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
            crop_img = img[y0:y2, x0:x2]
            __save_letter(d[letter], crop_img, 'dict')
            return d[letter]  # Mistake case
    except KeyError:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
        crop_img = img[y0:y2, x0:x2]
        __save_letter(letter, crop_img, 'unchanged')
        return letter  # Work well or unknown


def __instant_replace(letter):
    try:
        replace = {
            'j': 'ǰ',
            'J': 'ǰ',
            'x̌': 'ž',
            'ğ': 'š',
            'ś': 'ə' + u'\u0301'}
        letter = replace[letter]
        return letter
    except KeyError:
        return letter


def __save_letter(letter, crop_img, img_comment=None, folder='letters'):
    if DEBUG:
        if img_comment is None:
            img_comment = str(time.time())
        folder_code = str('_'.join([str(ord(i)) for i in letter]))
        folder_name = folder_code + f'_{letter}'
        try:
            if not os.path.exists(os.path.join(folder, folder_name)):
                os.makedirs(os.path.join(folder, folder_name))
            cv2.imwrite(os.path.join(folder, folder_name, f'{img_comment}.jpg'), crop_img)
        except OSError:
            if not os.path.exists(os.path.join(folder, folder_code)):
                os.makedirs(os.path.join(folder, folder_code))
            cv2.imwrite(os.path.join(folder, folder_code, f'{img_comment}.jpg'), crop_img)


def __under_dot(img, poly, letter):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    crop_img = img[y2 - 2:y2 + 18, x0:x2]
    crop_img = cv2.copyMakeBorder(crop_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, None, value=(230, 255, 255))

    template = cv2.imread('templates/dot.jpg')
    prob = np.max(cv2.matchTemplate(crop_img, template, cv2.TM_CCOEFF_NORMED))
    if prob > 0.45:
        letter = letter + u'\u0323'

    __save_letter(letter, img[y0:y2 + 18, x0:x2], 'udot')
    return letter


def __check_reversed_e(img, poly, letter):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    crop_img = img[y0:y2, x0:x2]
    bounded_img = cv2.copyMakeBorder(crop_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=(230, 255, 255))

    prob1 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/e.jpg'), cv2.TM_CCOEFF_NORMED))
    prob2 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/ə.jpg'), cv2.TM_CCOEFF_NORMED))

    if prob1 > prob2:
        letter = 'e'
    else:
        prob3 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/ə_acute.jpg'), cv2.TM_CCOEFF_NORMED))
        if prob3 > prob2:
            letter = 'ə́'
        else:
            letter = 'ə'

    __save_letter(letter, crop_img, 'revers_e')
    return letter


def __check_reversed_e_acute(img, poly, letter):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    crop_img = img[y0:y2, x0:x2]
    bounded_img = cv2.copyMakeBorder(crop_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=(230, 255, 255))

    prob1 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/e_acute.jpg'), cv2.TM_CCOEFF_NORMED))
    prob2 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/ə_acute.jpg'), cv2.TM_CCOEFF_NORMED))
    if prob1 > prob2:
        letter = 'é'
    else:
        letter = 'ə́'

    __save_letter(letter, crop_img, 'revers_e_acute')
    return letter


def __check_reversed_e_acute_o(img, poly, letter):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    crop_img = img[y0:y2, x0:x2]
    bounded_img = cv2.copyMakeBorder(crop_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=(230, 255, 255))

    prob1 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/o_acute.jpg'), cv2.TM_CCOEFF_NORMED))
    prob2 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/ə_acute.jpg'), cv2.TM_CCOEFF_NORMED))
    if prob1 > prob2:
        letter = 'ó'
    else:
        letter = 'ə́'

    __save_letter(letter, crop_img, 'revers_e_acute_o')
    return letter


def __acute_o(img, poly, letter):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    crop_img = img[y0:y2, x0:x2]
    bounded_img = cv2.copyMakeBorder(crop_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=(230, 255, 255))

    prob1 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/o_acute.jpg'), cv2.TM_CCOEFF_NORMED))
    prob2 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/o.jpg'), cv2.TM_CCOEFF_NORMED))
    if prob1 > prob2:
        letter = 'ó'
    else:
        letter = 'o'

    __save_letter(letter, crop_img, 'comb_o')
    return letter


def __acute_k(img, poly, letter):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    crop_img = img[y0:y2, x0:x2]
    bounded_img = cv2.copyMakeBorder(crop_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=(230, 255, 255))

    prob1 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/k_acute.jpg'), cv2.TM_CCOEFF_NORMED))
    prob2 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/k.jpg'), cv2.TM_CCOEFF_NORMED))
    if prob1 > prob2:
        letter = 'ḱ'
    else:
        letter = 'k'

    __save_letter(letter, crop_img, 'comb_k')
    return letter


def __acute_g(img, poly, letter):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    crop_img = img[y0:y2, x0:x2]
    bounded_img = cv2.copyMakeBorder(crop_img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, None, value=(230, 255, 255))

    prob1 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/g_acute.jpg'), cv2.TM_CCOEFF_NORMED))
    prob2 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/g.jpg'), cv2.TM_CCOEFF_NORMED))
    if prob1 > prob2:
        letter = 'ǵ'
    else:
        letter = 'g'

    __save_letter(letter, crop_img, 'comb_g')
    return letter


def __acute_i(img, poly, letter):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    crop_img = img[y0:y2, x0:x2]
    bounded_img = cv2.copyMakeBorder(crop_img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, None, value=(230, 255, 255))

    prob1 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/i_acute.jpg'), cv2.TM_CCOEFF_NORMED))
    prob2 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/i.jpg'), cv2.TM_CCOEFF_NORMED))
    if prob1 > prob2:
        letter = 'í'
    else:
        letter = 'i'

    __save_letter(letter, crop_img, 'comb_i')
    return letter


def __upper_comb_u(img, poly, letter):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    crop_img = img[y0 - 15:y2 - 25, x0:x2]
    bounded_img = cv2.copyMakeBorder(crop_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=(230, 255, 255))

    prob1 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/line_acute.jpg'), cv2.TM_CCOEFF_NORMED))
    prob2 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/line.jpg'), cv2.TM_CCOEFF_NORMED))

    if letter in ['u', 'ú', 'ū', 'ü']:
        letter = 'u'
    elif letter in ['a', 'á', 'ä', 'ã', 'ā']:
        letter = 'a'

    # print(prob1)
    if prob1 > 0.51:
        if prob1 > prob2:
            letter = 'ā́' if letter == 'a' else 'ū́'  # letter += u'\u0304' + u'\u0301'
        else:
            letter = 'ā' if letter == 'a' else 'ū'  # letter += u'\u0304'
    else:
        prob3 = np.average(crop_img) / 255.
        if prob3 > 0.95:  # Empty space
            if letter == 'a':
                crop_img = img[y0:y2, x0:x2]
                bounded_img = cv2.copyMakeBorder(crop_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None,
                                                 value=(230, 255, 255))
                prob5 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/ə.jpg'),
                                                 cv2.TM_CCOEFF_NORMED))
                if prob5 > 0.76:
                    letter = 'ə'
        else:
            letter = 'á' if letter == 'a' else 'ú'  # letter += u'\u0301'
            if letter == 'á':
                crop_img = img[y0:y2, x0:x2]
                bounded_img = cv2.copyMakeBorder(crop_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None,
                                                 value=(230, 255, 255))
                prob4 = np.max(cv2.matchTemplate(bounded_img, cv2.imread('templates/ə_acute.jpg'),
                                                 cv2.TM_CCOEFF_NORMED))
                if prob4 > 0.8:
                    letter = 'ə́'
                else:
                    letter = 'á'

    __save_letter(letter, img[y0 - 15:y2, x0:x2], 'comb_u')
    return letter
