# Marakulin Andrey https://github.com/Annndruha
# 2023
import os.path
import pickle
import glob

from google.cloud.vision import AnnotateImageResponse

from process_text import process_text


if __name__ == '__main__':
    # shutil.rmtree('letters', ignore_errors=True)




    folders = glob.glob('tests/*')
    for folder in folders:
        IMG_PATH = os.path.join(folder, "img.png")
        RESP_PATH = os.path.join(folder, "google_response.pickle")
        with open(RESP_PATH, "rb") as f:
            response: AnnotateImageResponse = pickle.load(f)
        text = process_text(IMG_PATH, response)
        with open(os.path.join(folder, "result.txt"), 'w') as f:
            f.write(text)
