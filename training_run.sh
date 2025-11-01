#!/bin/bash

# python3 live_training.py --episodes 4 --discount-factor 0.0 --train-every 1

python3 training.py --episodes 32 --discount-factor 0.0 --sample-size 14
python3 training.py --episodes 32 --discount-factor 0.1 --sample-size 14
python3 training.py --episodes 32 --discount-factor 0.2 --sample-size 14
python3 training.py --episodes 32 --discount-factor 0.3 --sample-size 14
python3 live_training.py --episodes 32 --discount-factor 0.5 --train-every 2
python3 live_training.py --episodes 1024 --discount-factor 0.99 --train-every 8
