#!/bin/bash
mkdir -p docker/.pyrec_dataset
if test -z "$(docker inspect --type=image pyrec:0.0.1)"; then
  docker build -t pyrec:0.0.1 -f docker/Dockerfile .
fi