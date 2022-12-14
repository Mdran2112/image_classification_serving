
import base64
import os

import cv2


def image_to_base64(image_np):
    image = cv2.imencode('.png', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]

    return image_code

def check_if_file_existis(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)