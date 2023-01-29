# Marakulin Andrey https://github.com/Annndruha
# 2023

import pickle

from google.cloud.vision import AnnotateImageResponse

from postprocess_text import postprocess_text


if __name__ == '__main__':
    IMG_PATH = "tests/pages_20-21/img.png"
    RESP_PATH = "tests/pages_20-21//google_response.pickle"
    with open(RESP_PATH, "rb") as f:
        response: AnnotateImageResponse = pickle.load(f)
    text = postprocess_text(IMG_PATH, response, 300000)
    with open(IMG_PATH.replace('img.png', 'result.txt'), 'w') as f:
        f.write(text)
