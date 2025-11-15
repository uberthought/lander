#!/bin/bash

python3 training.py --episodes 32 --discount-factor 0.0 --sample-size 15

# python3 live_training.py --episodes 32 --discount-factor 0.1 --train-every 4
python3 live_training.py --episodes 32 --discount-factor 0.5 --train-every 4
python3 live_training.py --episodes 32 --discount-factor 0.9 --train-every 4
# python3 live_training.py --episodes 32 --discount-factor 0.95 --train-every 4

while true; do
    python3 live_training.py --episodes 32 --discount-factor 0.99 --train-every 4
done
