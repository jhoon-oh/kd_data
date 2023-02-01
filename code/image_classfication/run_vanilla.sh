#!/bin/bash

# teacher is None

python main.py --device=cuda:0 \
               --student=wrn-28-2 \
               --dataset=cifar10 \
               --batch_size=128 \
               --num_epochs=200 \
               --mode=crop \
               --nesterov \
               --seed=9999
               
echo "finished"
