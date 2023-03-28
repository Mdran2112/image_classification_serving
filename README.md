# image_classification_serving
This is an example of REST API, built with Python and FastAPI, that can be used to serve a deep learning model for image classification trained with Keras (Tensorflow).
The API has two endpoints: 

#### POST  /classifier/predictions
Make predictions for a batch of images, using an image classification model. Requires a json body with this schema:
```
{
   "images": [
      {
         "image_id": 0,    // image identification [int]
         "img_base64": ...    // string of encoded image in base64 [str]
      },
      {
         "image_id": 1,
         "img_base64": ...
      },   
   ]
}
```


#### PUT   /classifier/prediction-service/{new-model}
It is used to load/change the classification model package and configure the prediction pipeline according to the model configuration file (see Model package). 


## Model package

A model package represents a group of several files, consumed by the prediction service for classification. 
In the model/ directory, there is an example of how a "model package" should be; it consists in:
* hdf5 model file (<model_name>.hdf5)
* json configuration file (<model_name>_config.json)

Those files have to be localted inside a folder named as the model itself. For example:
```
my_model/
        |__my_model.hdf5
        |__ my_model_config.json
```
 Note that the json config filename needs the sufix ```_config```.
 The folder ```my_model/``` will be inside the docker ```models``` volume (see docker-compose.yml).

## Prediction Service

The Prediction Service uses the model json configuration file in order to configure how image preprocessing and output processing will be. With that information, it creates a prediction pipeline (decoding and parsing images + preprocessing + predict + output processing) and delivers a response with the results.
            
## Build & deploy

This repository has a Dockerfile for building a docker image. By executing ```docker_build.sh```, a docker image will be created.
The deploy can be made by using the ```docker-compose.yml``` file. The ```X_API_KEY``` environment variable is for authentication when using the API (see the example in the demo jupyter notebook.)
The model packages (folders with models and json configs) have to be inside the ```models/``` volume.

## Demo notebook.

Inside the ```demo/``` directory there is a Jupyter notebook. The first cell (after imports) generates a batch of images transformed to base64 and builds an input json body. Then, send a request to ```/classifier/prediction-service/{new-model}``` in order loads the example classification model (it has a pretty bad accuracy... but it is useful for testing) with its pipeline configuration. Finally send a request to ```/classifier/predictions``` with the json body.


