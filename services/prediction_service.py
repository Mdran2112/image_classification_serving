import json
import pathlib
from typing import List, Dict, Any
import numpy as np
import os
import gc
import tensorflow.keras as ks
from numpy import ndarray

from modules.image_preprocessor.img_parser import ImageParser
from modules.image_preprocessor.img_preproc_factory import ImagePreprocessorFactory
from modules.output_postprocessor.output_postproc_factory import OutputPostprocessorFactory
from modules.output_postprocessor.output_preproc import Prediction
from services.utils import check_if_file_existis


class PredictionService:

    def __init__(self, model_name: str) -> None:

        models_base_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "models")
        self.model_name = model_name
        model_hdf5_path = os.path.join(models_base_path, model_name, f"{model_name}.hdf5")
        model_config_path = os.path.join(models_base_path, model_name, f"{model_name}_config.json")

        check_if_file_existis(model_hdf5_path)
        check_if_file_existis(model_config_path)

        self.model = ks.models.load_model(model_hdf5_path)

        with open(model_config_path, 'r') as jfile:
            config = json.load(jfile)
            jfile.close()

        self.image_preprocessor = ImagePreprocessorFactory.get(config["img_preproc_type"],
                                                               **config["image_preproc_config"])
        self.output_processor = OutputPostprocessorFactory.get(config["output_postproc_type"],
                                                               **config["output_config"])

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
                           scores=softmax)
            pred_list.append(p)
        return pred_list

    def _predict_proc(self, preds: List[Prediction]) -> List[Prediction]:
        return list(map(self.output_processor.do, preds))

    def _ok_response(self, pred_list: List[Prediction]) -> Dict[str, Any]:

        return {
            "code": 200,
            "response": {
                "model": self.model_name,
                "results": [res.to_json() for res in pred_list]
            }
        }
