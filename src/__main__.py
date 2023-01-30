# Marakulin Andrey https://github.com/Annndruha
# 2023
import os.path
import pickle
import glob

from google.cloud.vision import AnnotateImageResponse

from postprocess_text import postprocess_text


if __name__ == '__main__':
    folders = glob.glob('tests/*')
    for folder in folders:
        IMG_PATH = os.path.join(folder, "img.png")
        RESP_PATH = os.path.join(folder, "google_response.pickle")
        with open(RESP_PATH, "rb") as f:
            response: AnnotateImageResponse = pickle.load(f)
        text = postprocess_text(IMG_PATH, response, debug=True)
        with open(os.path.join(folder, "result.txt"), 'w') as f:
            f.write(text)
        # print(f'Done: {folder}')
