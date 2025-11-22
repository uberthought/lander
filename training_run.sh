#!/bin/bash

python3 training.py --episodes 32 --discount-factor 0.0 --sample-size 16

python3 live_training.py --episodes 16 --discount-factor 0.0 --train-every 4
python3 live_training.py --episodes 16 --discount-factor 0.1 --train-every 4
python3 live_training.py --episodes 16 --discount-factor 0.2 --train-every 4
python3 live_training.py --episodes 16 --discount-factor 0.3 --train-every 4
python3 live_training.py --episodes 16 --discount-factor 0.4 --train-every 4
python3 live_training.py --episodes 16 --discount-factor 0.5 --train-every 4
python3 live_training.py --episodes 16 --discount-factor 0.6 --train-every 4
python3 live_training.py --episodes 16 --discount-factor 0.7 --train-every 4
python3 live_training.py --episodes 16 --discount-factor 0.8 --train-every 4

while true; do
    python3 live_training.py --episodes 64 --discount-factor 0.9 --train-every 4
done
