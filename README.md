# Provable-Benefit-Local-Steps-Feature-Learning
This is a repo for ICML 2024 paper: Provable Benefits of Local Steps in Heterogeneous Federated Learning for Neural Networks: A Feature Learning Perspective

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
python3 visualize_resnset.py path/to/logs/
```
where `path/to/logs` should be replaced with the directory storing results for the experiment whose network should be visualized.
