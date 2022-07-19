#!/bin/bash -e
IMAGE_NAME="dajobe/hbase"
CONTAINER_NAME="hbase-docker"

docker stop "${CONTAINER_NAME}"
