#!/bin/bash

# python3 training.py --episodes 32 --discount-factor 0.0 --sample-size 17

# python3 live_training.py --episodes 32 --discount-factor 0.1 --train-every 4
# python3 live_training.py --episodes 32 --discount-factor 0.3 --train-every 4
# python3 live_training.py --episodes 32 --discount-factor 0.5 --train-every 4
# python3 live_training.py --episodes 32 --discount-factor 0.6 --train-every 4
# python3 live_training.py --episodes 32 --discount-factor 0.7 --train-every 3
# python3 live_training.py --episodes 32 --discount-factor 0.8 --train-every 3
python3 live_training.py --episodes 32 --discount-factor 0.9 --train-every 2
python3 live_training.py --episodes 32 --discount-factor 0.95 --train-every 2

while true; do
    python3 live_training.py --episodes 32 --discount-factor 0.97 --train-every 4
done
