#!/bin/bash

python3 live_training.py --episodes 32 --discount-factor 0.0 --train-every 1

python3 training.py --episodes 32 --discount-factor 0.0 --sample-size 17
python3 live_training.py --episodes 8 --discount-factor 0.1 --train-every 1
python3 live_training.py --episodes 8 --discount-factor 0.3 --train-every 1
python3 live_training.py --episodes 16 --discount-factor 0.5 --train-every 1
python3 live_training.py --episodes 16 --discount-factor 0.9 --train-every 1

while true; do
    python3 live_training.py --episodes 32 --discount-factor 0.99 --train-every 8
done
