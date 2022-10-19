from typing import List

from pydantic import BaseModel


class ImgModel(BaseModel):
    img_base64: str
    image_id: int


class ImgPredModelBody(BaseModel):
    images: List[ImgModel]
