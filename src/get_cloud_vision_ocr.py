# Marakulin Andrey https://github.com/Annndruha
# 2023

import io
import pickle

from google.cloud import vision


def get_response(path):
    """Detects text in the file."""

    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return response


if __name__ == '__main__':
    # HOW TO ADD Default Credentials
    # https://cloud.google.com/vision/docs/detect-labels-image-client-libraries
    # If you use PyCharm set in into configurations environment variable
    # For example:
    # GOOGLE_APPLICATION_CREDENTIALS=C:\Users\%username%\google_credentials.json

    resp = get_response("tests/page149/img.png")
    with open("tests/page149/google_response.pickle", "wb") as f:
        pickle.dump(resp, f)

    # with open("tests/text1/google_response.pickle", "rb") as f:
    #     resp_reconstructed = pickle.load(f)
    #     print(resp_reconstructed)
