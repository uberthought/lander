#!/bin/bash

python3 training.py --episodes 32 --discount-factor 0.0 --sample-size 20

python3 live_training.py --episodes 32 --discount-factor 0.0 --train-every 2
python3 live_training.py --episodes 32 --discount-factor 0.2 --train-every 2
python3 live_training.py --episodes 32 --discount-factor 0.4 --train-every 2
python3 live_training.py --episodes 32 --discount-factor 0.6 --train-every 2
python3 live_training.py --episodes 32 --discount-factor 0.8 --train-every 2

# python3 training.py --episodes 2 --discount-factor 0.95 --sample-size 22

while true; do
    python3 live_training.py --episodes 32 --discount-factor 0.95 --train-every 2
done
