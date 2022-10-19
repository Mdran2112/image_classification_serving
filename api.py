import json

import uvicorn
from fastapi import FastAPI
from fastapi.security import APIKeyHeader
from fastapi import Security

from decorators import authorize, handle_error, X_API_KEY
from schemas.pred_service_models import ImgPredModelBody
from services.prediction_service import PredictionService
import tensorflow.keras as ks
import gc

PREDICTION_SERVICE = PredictionService(model_name="animals_classifier_v0")

DOCS_URL = f"/docs/{X_API_KEY}"
REDOC_URL = f"/redoc/{X_API_KEY}"

app = FastAPI(
    title="Image Classification Server",
    description="REST API for image classification model serving.",
    version="v0",
    contact={
        "name": "Martín D.",
        "email": "dran2112@gmail.com",
    },
    docs_url=DOCS_URL,
    redoc_url=REDOC_URL)

URL_PREFIX = "/classifier"
api_key_header = APIKeyHeader(name='X-API-Key', auto_error=True)


@app.post(f"{URL_PREFIX}/predictions")
@authorize
@handle_error
def predictions(request_body: ImgPredModelBody, header: str = Security(api_key_header)):
    global PREDICTION_SERVICE

    body = list(map(lambda x: json.loads(x.json()), request_body.images))
    ks.backend.clear_session()
    resp = PREDICTION_SERVICE.predict(body)
    return resp


@app.put(f"{URL_PREFIX}/change/{{model_name}}")
@authorize
@handle_error
def change(model_name: str, header: str = Security(api_key_header)):
    global PREDICTION_SERVICE

    ks.backend.clear_session()
    PREDICTION_SERVICE = PredictionService(model_name)
    gc.collect()
    resp = {
        "code": 200,
        "response": f"Changed model to {model_name}"
    }

    return resp


def run_api():
    uvicorn.run(
        app,
        port=8000,
        host="0.0.0.0",
        loop='asyncio'
    )

if __name__ == "__main__":
    run_api()
