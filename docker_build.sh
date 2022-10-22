#!/bin/bash

docker build --tag=model-serving-api .
docker image prune -f