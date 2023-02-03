# Marakulin Andrey https://github.com/Annndruha
# 2023
import os
import sys
import glob
import pickle
import argparse
from pathlib import Path

from google.cloud.vision import AnnotateImageResponse

from detector.process_text import process_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Google cloud vision text detection")
    parser.add_argument("imagepath", type=str,
                        help="Path to image for text detection")
    parser.add_argument("picklepath", type=str, default=None, nargs='?',
                        help="Path to associated pickle file (by google_ocr.py). "
                             "If None, use same filename as image, but with .pickle extension")
    cliargs = parser.parse_args()

    if cliargs.picklepath is None:
        p = Path(cliargs.imagepath)
        picklepath = p.with_suffix('.pickle')
        # print('None', cliargs.picklepath)
    else:
        picklepath = Path(cliargs.picklepath)
        # print(type(cliargs.picklepath), cliargs.picklepath)

    with open(picklepath, "rb") as f:
        response: AnnotateImageResponse = pickle.load(f)

    text = process_text(cliargs.imagepath, response)

    with open(picklepath.with_suffix('.txt'), 'w') as f:
        f.write(text)
        print('Result saved as', picklepath.with_suffix('.txt'))

    # folders = glob.glob('tests/*')
    # for folder in folders:
    #     IMG_PATH = os.path.join(folder, "img.png")
    #     RESP_PATH = os.path.join(folder, "google_response.pickle")
    #     with open(RESP_PATH, "rb") as f:
    #         response: AnnotateImageResponse = pickle.load(f)
    #     text = process_text(IMG_PATH, response)
    #     with open(os.path.join(folder, "result.txt"), 'w') as f:
    #         f.write(text)
