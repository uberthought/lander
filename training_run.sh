#!/bin/bash

# python3 live_training.py --episodes 256 --discount-factor 0.0 --train-every 256

python3 training.py --episodes 32 --discount-factor 0.0 --sample-size 20
# python3 live_training.py --episodes 32 --discount-factor 0.5 --train-every 2
# python3 live_training.py --episodes 32 --discount-factor 0.9 --train-every 2
while true; do
    python3 live_training.py --episodes 64 --discount-factor 0.9 --train-every 8
done
