from dataclasses import dataclass
from typing import Tuple, Callable, List

import cv2
import numpy as np


class ImagePreprocessor:

    def __init__(self, **kwargs):
        self.preproc_list: List[Callable[[np.ndarray], np.ndarray]] = []

    def do(self, img: np.ndarray) -> np.ndarray:
        for preproc in self.preproc_list:
            img = preproc(img)
        return img


@dataclass
class NormAndPaddingImagePreprocessor(ImagePreprocessor):
    input_shape: Tuple[int, int, int]
    normalization_factor: int = 255

    def __post_init__(self):
        super().__init__()
        self.preproc_list: List[Callable[[np.ndarray], np.ndarray]] = [
            self._add_padding,
            self._normalize
        ]

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        return img / self.normalization_factor

    def _add_padding(self, img: np.ndarray) -> np.ndarray:
        height, width, _ = img.shape

        new_width, new_height = self.input_shape[1], self.input_shape[0]

        left = (new_width - width) / 2
        top = (new_height - height) / 2
        right = np.ceil((new_width - width) / 2)
        bottom = np.ceil((new_height - height) / 2)

        im = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
        return im.astype("uint8")


class NoPreproc(ImagePreprocessor):
    ...
