from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


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
class OutputProcessor:
    output_classes: List[str]

    def _map_label(self, pred: Prediction) -> Prediction:
        pred.label_str = self.output_classes[pred.max_score_inx]
        return pred

    def do(self, pred: Prediction) -> Prediction:
        pred = self._map_label(pred)
        return pred
