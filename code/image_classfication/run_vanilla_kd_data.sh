#!/bin/bash

python main.py --device=cuda:0 \
               --teacher=wrn-28-4 \
               --student=wrn-16-2 \
               --dataset=cifar100 \
               --batch_size=128 \
               --num_epochs=200 \
               --mode=crop \
               --nesterov \
               --alpha=1.0 \
               --temperature=20.0 \
               --delta=0.1 \
               --zeta=1.0 \
               --seed=9999
               
echo "finished"
