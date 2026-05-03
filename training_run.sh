#!/bin/bash

python3 collect_random_data.py --episodes 128
# python3 training.py --seconds 30 --sample-size 16

while true; do
    python3 live_training.py --seconds 30
    # python3 world_training.py --seconds 30
done