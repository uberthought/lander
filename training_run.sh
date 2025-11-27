#!/bin/bash

python3 live_training.py --episodes 32 --discount-factor 0.0 --train-every 2
python3 live_training.py --episodes 32 --discount-factor 0.2 --train-every 2
python3 live_training.py --episodes 32 --discount-factor 0.4 --train-every 2
python3 live_training.py --episodes 32 --discount-factor 0.6 --train-every 2
python3 live_training.py --episodes 32 --discount-factor 0.8 --train-every 2


while true; do
    python3 live_training.py --episodes 32 --discount-factor 0.9 --train-every 2
done
