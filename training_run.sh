#!/bin/bash

python3 collect_random_data.py --episodes 128
python3 training.py --seconds 60 --sample-size 16
python3 live_training.py --seconds 60
