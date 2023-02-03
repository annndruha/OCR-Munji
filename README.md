## OCR-Munji

Text detection of printed text in Munji language.

Detector created for book "Грюнберг А.Л. — Мунджанский язык Тексты"

![readme.png](readme.png)

### Alghoritm

The detector is based on Google cloud vision text detection with additional heuristics that recognize the characters of Munji language. A variety of heuristics are used, such as the correlation of special characters or signs and the replacement of some letters obtained by Google text detection. (See `detector/mapping.py`)

### Using 

#### Step 1

For use Google cloud vision you need to get [GOOGLE_APPLICATION_CREDENTIALS](https://cloud.google.com/vision/docs/detect-labels-image-client-libraries#before-you-begin) and set corresponding environment variable.

#### Step 2

Get Google cloud vision text detectiob response for image:
```commandline
python src\google_ocr.py --image img.png --print-result
```
If command succeed, response saved as `.pickle` file.

#### Step 3
