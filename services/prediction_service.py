import base64
import json
import pathlib
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any, Callable
import numpy as np
import cv2
import os
import gc
import tensorflow.keras as ks
from numpy import ndarray


@dataclass
class Prediction:
    image_id: int
    scores: List[float]
    base64_img: str
    label_str: Optional[str] = None
    max_score_inx: Optional[int] = None

    def __post_init__(self):
        inx = int(np.argmax(self.scores))
        self.max_score = float(self.scores[inx])
        self.max_score_inx = int(inx)

    def to_json(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "max_score": self.max_score,
            "label": self.label_str
        }


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


@dataclass
class ImagePreprocessor:
    input_model_shape: Tuple[int, int, int]
    norm_factor: int

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


@dataclass
class OutputProcessor:
    output_classes: List[str]

    def _map_label(self, pred: Prediction) -> Prediction:
        pred.label_str = self.output_classes[pred.max_score_inx]
        return pred

    def do(self, pred: Prediction) -> Prediction:
        pred = self._map_label(pred)
        return pred


class PredictionService:

    def __init__(self, model_name: str):
        models_base_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "models")
        self.model_name = model_name
        self.model = ks.models.load_model(models_base_path + f"/{model_name}/{model_name}.hdf5")

        with open(os.path.join(models_base_path, model_name, f"{model_name}_config.json"), 'r') as jfile:
            config = json.load(jfile)

        self.image_preprocessor = ImagePreprocessor(config["image_preproc_config"]["input_shape"],
                                                    config["image_preproc_config"]["normalization_factor"])
        self.output_processor = OutputProcessor(config["output_classes"])
        self.image_parser = ImageParser()

    def predict(self, request_body: List[Dict[str, Any]]) -> Dict[str, Any]:
        pred_list = self._pipeline(request_body)
        resp = self._ok_response(pred_list)
        gc.collect()
        return resp

    def _pipeline(self, request_body: List[Dict[str, Any]]) -> List[Prediction]:
        img_array_list = self._parse_images(request_body)
        img_array_list = self._preproc(img_array_list)
        input_ = np.array(img_array_list)
        predictions_softmax = self.model.predict(input_)
        pred_list = self._parse_results(request_body, predictions_softmax)
        pred_list = self._predict_proc(pred_list)
        return pred_list

    def _parse_images(self, request_body: List[Dict[str, Any]]) -> List[ndarray]:
        return list(map(lambda x: self.image_parser.parse(x["img_base64"]), request_body))

    def _preproc(self, img_array_list: List[np.ndarray]) -> List[np.ndarray]:
        img_array_list = list(map(self.image_preprocessor.do, img_array_list))

        return img_array_list

    @staticmethod
    def _parse_results(request_body: List[Dict[str, Any]],
                       predictions_softmax: List[List[float]]) -> List[Prediction]:

        pred_list = []

        for img, softmax in zip(request_body, predictions_softmax):
            p = Prediction(image_id=img["image_id"],
                           scores=softmax,
                           base64_img=img["img_base64"])
            pred_list.append(p)
        return pred_list

    def _predict_proc(self, preds: List[Prediction]) -> List[Prediction]:
        return list(map(self.output_processor.do, preds))

    def _ok_response(self, pred_list: List[Prediction]) -> Dict[str, Any]:
        res = []
        for r in pred_list:
            res.append(r.to_json())
        return {
            "code": 200,
            "response": {
                "model": self.model_name,
                "results": res
            }
        }

