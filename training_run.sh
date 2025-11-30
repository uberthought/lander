#!/bin/bash

python3 collect_random_data.py --episodes 1024
python3 training.py --seconds   180
python3 live_training.py
