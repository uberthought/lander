#!/bin/bash

# python3 training.py --episodes 32 --discount-factor 0.99 --sample-size 22

while true; do
    python3 live_training.py --episodes 32 --discount-factor 0.99 --train-every 2
done
