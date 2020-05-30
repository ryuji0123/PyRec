#!/bin/bash
mkdir -p docker/.pyrec_dataset
if test -z "$(docker images -q pyrec:0.0.2)" -o -n "$(git diff --exit-code requirements_dev.txt)"; then
  dpcker system prune -f
  docker build -t pyrec:0.0.2 -f docker/Dockerfile .
fi