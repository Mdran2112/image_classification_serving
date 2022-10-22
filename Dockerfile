FROM python:3.8.10-slim

# set lang
ENV LANG C.UTF-8

# Set the working directory to /app
WORKDIR /app

COPY requirements.txt /app

RUN apt-get update

# install
RUN pip install -r requirements.txt

# install these libraries in order to get OpenCV work.
RUN apt install -y libsm6 libglib2.0-0 libxrender1 libxext6

COPY . /app

RUN \
    rm -r .git \
    && rm .gitignore \
    && rm docker-compose.yml \
    && rm docker_build.sh \
    && rm Dockerfile \
    && rm -r demo \
    && rm -r models

CMD ["python", "api.py"]
