#!/bin/bash

# teacher is None

python main.py --device=cuda:0 \
               --student=wrn-28-4 \
               --dataset=cifar100 \
               --batch_size=128 \
               --num_epochs=200 \
               --mode=crop \
               --nesterov \
               --seed=9999
               
echo "finished"
