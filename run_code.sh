#!/bin/bash

# Run TD3.py 3 times
for i in {1..3}; do
  echo "Running TD3.py, iteration $i"
  python TD3_V2X_V4.py
done

# Run SAC.py 3 times
for i in {1..3}; do
  echo "Running SAC.py, iteration $i"
  python SAC4.py
done

# Run TRPO.py 3 times
for i in {1..3}; do
  echo "Running TRPO.py, iteration $i"
  python3 TRPObufferv31.py
done

# Run DDPG.py 3 times
for i in {1..3}; do
  echo "Running DDPG.py, iteration $i"
  python ddpg.py
done
