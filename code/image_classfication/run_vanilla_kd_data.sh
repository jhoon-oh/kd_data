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
               --cls_acq=random \
               --cls_lower_qnt=0.0 \
               --cls_upper_qnt=0.1 \
               --per_class \
               --seed=9999

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
               --sample_acq=random \
               --sample_lower_qnt=0.0 \
               --sample_upper_qnt=0.1 \
               --per_class \
               --seed=9999

echo "finished"
