#!/usr/bin/env bash

CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/get_flops.py $CONFIG ${@:2}
