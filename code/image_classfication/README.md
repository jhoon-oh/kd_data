# Paper Title

This repository is the official implementation of "When does Knowledge Distillation Help?" for the image classification tasks.


## Training
Experiments in our paper can be reproduced as the following .sh files.

1. To get a teacher model, firstly run this command:

```train
./run_vanilla.sh
```

You can get other baseline models by changing `--student` argument. Also, you can also get the baseline model on CIFAR10 by changing `--dataset` arguments to `cifar10`.


2. To get a distilled student model, run this command:
```
./run_vanilla_kd.sh
```
Here, you can handle the amount of distillation by changing both `--alpha` and `--temperature`. If you use `--temperature` larger than 100, then we replace the gradient of the logit vector with a gradient that converges when the `temperature` goes to infinity.

3. To handle the amount of training data, run this command:
```
./run_vanilla_kd_data.sh
```
In this file, the `(delta, zeta)` is set to (0.1, 1.0). Detailed explanations are described in `main.py`.

## Evaluation

To evaluate the model(s) and see the results via the entropy or TLD, please refer to the `grid_plot.ipynb` and `pdf_plot.ipynb`.


## Results
All our results can be reproduced by our code. All results have been described in our paper including Appendix. The results of our experiments are so numerous that it is difficult to post everything here. However, if you experiment several times by modifying the hyperparameter value in the .sh file, you will be able to reproduce all of our analyzes.


## Dependencies
Our code is tested under the following environment:
1. Python 3.7.3
2. PyTorch 1.1.0
3. torchvision 0.3.0
4. Numpy 1.16.2
5. tqdm