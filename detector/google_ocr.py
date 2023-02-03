# Marakulin Andrey https://github.com/Annndruha
# 2023

import io
import pickle
import argparse
from pathlib import Path

from google.cloud import vision
from google.cloud.vision import AnnotateImageResponse


def get_response(img_path):
    """Detects text in the file."""

    client = vision.ImageAnnotatorClient()

    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    resp = client.text_detection(image=image)

    if resp.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                resp.error.message))

    return resp


if __name__ == '__main__':
    # HOW TO ADD Default Credentials
    # https://cloud.google.com/vision/docs/detect-labels-image-client-libraries
    # If you use PyCharm set in into configurations environment variable
    # For example:
    # GOOGLE_APPLICATION_CREDENTIALS=C:\Users\%username%\google_credentials.json

    parser = argparse.ArgumentParser(description="Google cloud vision text detection")
    parser.add_argument("--path", default=None,
                        help="Path to image for text detection")
    parser.add_argument("--print-result", action="store_true",
                        help="Print detection result as json")
    parser.add_argument("--save-json", action="store_true",
                        help="Save json instead of pickle")
    cliargs = parser.parse_args()

    response = get_response(cliargs.path)
    path = Path(cliargs.path)

    if cliargs.print_result:
        print(response)

    if cliargs.save_json:
        with open(path.with_suffix('.json'), 'w') as f:
            json = AnnotateImageResponse.to_json(response)
            f.write(json)
            print('Google cloud vision text detection response saved as', path.with_suffix('.json'))
    else:
        with open(path.with_suffix('.pickle'), "wb") as f:
            pickle.dump(response, f)
            print('Google cloud vision text detection response saved as', path.with_suffix('.pickle'))
