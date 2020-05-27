FROM python:3.8
USER root
COPY . /PyRec
WORKDIR /PyRec

RUN apt-get update && \
    pip install --upgrade pip && \
    pip install -r requirements_dev.txt