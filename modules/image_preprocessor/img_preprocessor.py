from dataclasses import dataclass
from typing import Tuple, Callable, List

import cv2
import numpy as np

@dataclass
class ImagePreprocessor:
    input_model_shape: Tuple[int, int, int]
    norm_factor: int = 255

    def __post_init__(self):
        self.preproc_list: List[Callable[[np.ndarray], np.ndarray]] = [
            self._add_padding,
            self._normalize
        ]

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        return img / self.norm_factor

    def _add_padding(self, img: np.ndarray) -> np.ndarray:
        height, width, _ = img.shape

        new_width, new_height = self.input_model_shape[1], self.input_model_shape[0]

        left = (new_width - width) / 2
        top = (new_height - height) / 2
        right = np.ceil((new_width - width) / 2)
        bottom = np.ceil((new_height - height) / 2)

        im = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
        return im.astype("uint8")

    def do(self, img: np.ndarray) -> np.ndarray:
        for preproc in self.preproc_list:
            img = preproc(img)
        return img

