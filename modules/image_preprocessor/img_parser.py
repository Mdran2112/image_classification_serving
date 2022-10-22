import base64
from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class ImageParser:

    @classmethod
    def parse(cls, base64_code: str) -> np.ndarray:
        return cls.base64_to_image(base64_code)

    @classmethod
    def base64_to_image(cls, base64_code: str):
        img_data = base64.b64decode(base64_code)
        img_array = np.fromstring(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)

        return img



