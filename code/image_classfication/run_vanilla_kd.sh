#!/bin/bash
temp="1.0 3.0 5.0"
for tau in $temp
do
    python main.py --device=cuda:0 \
                   --teacher=wrn-28-4 \
                   --student=wrn-28-4 \
                   --dataset=cifar10 \
                   --batch_size=128 \
                   --num_epochs=200 \
                   --mode=crop \
                   --nesterov \
                   --alpha=1.0 \
                   --temperature=$tau \
                   --seed=9999
done
echo "finished"
