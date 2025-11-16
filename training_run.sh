#!/bin/bash

python3 training.py --episodes 32 --discount-factor 0.0

python3 live_training.py --episodes 32 --discount-factor 0.5 --train-every 2

while true; do
    python3 live_training.py --episodes 32 --discount-factor 0.9 --train-every 2
done
