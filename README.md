# image_classification_serving
This is an example of REST API, made with Python and FastAPI, that can be used to serve a deep learning model for image classification trained with Keras (Tensorflow).
The API has two endpoints: 

#### POST  /classifier/predictions
Make predictions for a batch of images, using am image classification model. Requires a json body with this schema:
```
{
   "images": [
      {
         "image_id": 0,    // image identification [int]
         "img_base64": ...    // string of encoded image in base64 [int]
      },
      {
         "image_id": 1,
         "img_base64": ...
      },   
   ]
}
```


#### PUT   /classifier/prediction-service/{new-model}
It used to load/change the classification model package and configure the prediction pipeline according the model configuration. 


## Model package

A model package is an object that is consumed by the prediction service for classification. 
In the model/ directory, there is an example of how a "model package" should be; it consists in:
* hdf5 model file (<model_name>.hdf5)
* json configuration file (<model_name>_config.json)

Those files have to be localted inside a folder named as the model itself:
```
<model_name>/
            |__<model_name>.hdf5
            |__ <model_name>_config.json
```
 The folder <model_name>/ will be inside the docker models volume (see docker-compose.yml).

## Prediction Service

The Prediction Service uses the model json configuration file in order to configure how image preprocessing and output processing will be. With that information, 
it creates a prediction pipeline (parsing images + preprocessing + predict + output processing) and delivers a response with the results.
            
## Build & deploy

This repository has a Dockerfile for building a docker image. By executing ```docker_build.sh```, a docker image will be created.
The deploy can be made by using the docker-compose.yml file. The ```X_API_KEY``` environment variable is for authentication when using the API.
The model packages (folders with models and json configs) have to be inside the models/ volume.

## Demo notebook.

Inside the demo/ directory there is a Jupyter notebook, which generates a batch of images (encoded in base64) and send requests to the API in order
to get predictions. The demo loads an example classification model (with pretty bad accuracy... but it is useful for testing.) 
You can test the endpoints using that notebook as an example.

