# Provable-Benefit-Local-Steps-Feature-Learning
This is a repo for ICML 2024 paper: Provable Benefits of Local Steps in Heterogeneous Federated Learning for Neural Networks: A Feature Learning Perspective

## Background

Our work is motivated by an experimental finding of the CIFAR-10 task in the heterogeneous FL environment, where we can see that Local SGD learned more distinguishing features than parallel SGD in the following figure (e.g., one can compare the pixels inside the bounding boxes). This finding inspires us to analyze the benefits of local steps from a feature learning (Allen-Zhu & Li, 2022a;b)  perspective.

![](https://github.com/MingruiLiu-ML-Lab/Provable-Benefit-Local-Steps-Feature-Learning/blob/main/feature-comparison.png)


Our synthetic data model relies on the feature noise components in feature learning. To evaluate Local SGD under real-world data with feature noise, we train with modified CIFAR-10 data that explicitly includes feature noise similar to our theoretical framework. Examples of the modified images are shown in the following figure. Each row shows a modified image for a different value of the feature noise magnitude $\rho$, ranging over $\rho \in \{0.0, 0.03125, 0.0625, 0.125, 0.25\}$. Each image contains slightly more noise than the previous row, but overall the images retain their original signal.

![](https://github.com/MingruiLiu-ML-Lab/Provable-Benefit-Local-Steps-Feature-Learning/blob/main/noisy_images.png)


## Installation

The only requirement is PyTorch. You can follow [PyTorch's instructions](https://pytorch.org/get-started/previous-versions/) to install PyTorch however you like, but we have only tested in this code in a Conda environment with PyTorch 1.12.1 and CUDA 11.3, which can be installed with the following command:
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

## Running the Code
Experiments from the paper can be reproduced by navigating to `scripts/` and running
```
bash cifar10.sh
```
and
```
bash cifar10_feature_noise.sh
```
The experiments from the main body and Section F.1 are implemented in `cifar10.sh`, and the experiments from Section F.2 are implemented in `cifar10_feature_noise.sh`. Note that these scripts are expecting to run on a single node with 8 GPUs, but this can be modified by changing the variable `gpus` in each script from 8 to however many GPUs you have.

Results will be stored in the `logs/` directory, and can be plotted with `scripts/plot.py`.

The weight visualization from Figure 1 can be reproduced by running
```
python3 visualize_resnet.py path/to/logs/
```
where `path/to/logs` should be replaced with the directory storing results for the experiment whose network should be visualized.


## Citation
If you found this repository helpful, please cite our paper:
```
@inproceedings{bao2024provable,
title={Provable Benefits of Local Steps in Heterogeneous Federated Learning for Neural Networks: A Feature Learning Perspective},
author={Yajie Bao, Michael Crawshaw, Mingrui Liu},
booktitle={International Conference on Machine Learning},
year={2024}
}

```
