import os
import glob
import argparse
from math import ceil, sqrt

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18


SMALL = False


if SMALL:
    num_filters = 16
    selected = torch.randperm(num_filters)
else:
    num_filters = 64


def plot_features(weight_dir, state_dict_path):

    # Load in weights.
    net = resnet18()
    net.load_state_dict(torch.load(state_dict_path), strict=False)

    # Plot convolutional filters of first layer as images.
    grid_len = ceil(sqrt(num_filters))
    plt.figure(figsize=(grid_len, grid_len))
    gs = gridspec.GridSpec(grid_len, grid_len)
    gs.update(wspace=0, hspace=0)
    for i in range(num_filters):
        ax = plt.subplot(gs[i])
        j = int(selected[i]) if SMALL else i
        kernel = F.sigmoid(20 * net.conv1.weight[j].detach()).cpu().numpy()
        kernel = np.transpose(kernel, (1, 2, 0))
        ax.imshow(kernel)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    name = os.path.basename(os.path.dirname(state_dict_path))
    plt.tight_layout()
    plt.savefig(os.path.join(weight_dir, f"{name}_visualize.pdf"))
    plt.close()


def main(parent_dir):

    weight_dir = os.path.join(parent_dir, "weight_visualizations")
    if not os.path.isdir(weight_dir):
        os.makedirs(weight_dir)
    state_dict_paths = glob.glob(os.path.join(parent_dir, "*", "trained_model.pth"))
    for state_dict_path in state_dict_paths:
        plot_features(weight_dir, state_dict_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dirs", nargs="*", type=str)
    args = parser.parse_args()
    for parent_dir in args.parent_dirs:
        main(parent_dir)
