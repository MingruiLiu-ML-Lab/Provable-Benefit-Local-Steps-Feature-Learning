import os
from math import log2, ceil

import torch
import torch.nn as nn
import torchshow as ts


class SimpleCNN(nn.Module):
    """ Simple CNN architecture. """

    def __init__(self, dataset, num_layers, mod_last_layer=None, binarize_classes=False):
        super(SimpleCNN, self).__init__()

        assert num_layers > 0
        self.dataset = dataset
        self.num_layers = num_layers
        self.mod_last_layer = mod_last_layer
        self.binarize_classes = binarize_classes

        if self.dataset == "MNIST":
            w, h, c = (28, 28, 1)
            num_classes = 10
            init_channels = 10
        elif self.dataset == "FEMNIST":
            w, h, c = (28, 28, 1)
            num_classes = 62
            init_channels = 20
        elif self.dataset == "CIFAR10":
            w, h, c = (32, 32, 3)
            num_classes = 10
            init_channels = 40
        elif self.dataset == "CelebA":
            w, h, c = (64, 64, 3)
            num_classes = 2
            init_channels = 40
        else:
            raise NotImplementedError

        layers = []
        channels = c
        for i in range(self.num_layers):
            if i == 0:
                out_channels = init_channels
                stride = 2
                w = (w+1) // 2
                h = (h+1) // 2
            else:
                downsample = i % 2 == 0
                if downsample:
                    out_channels = channels * 2
                    stride = 2
                    w = (w+1) // 2
                    h = (h+1) // 2
                else:
                    out_channels = channels
                    stride = 1

            layers.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=channels, out_channels=out_channels, kernel_size=5, stride=stride, padding=2
                ),
                nn.ReLU(),
            ))
            channels = out_channels

        self.conv_layers = nn.ModuleList(layers)

        num_downsamples = (num_layers - 1) // 2
        feature_channels = init_channels * 2 ** num_downsamples
        feature_size = w * h * feature_channels
        out_size = 2 if binarize_classes else num_classes
        self.fc = nn.Linear(feature_size, out_size)

        if self.mod_last_layer == "fixed":
            assert feature_channels % out_size == 0
            features_per_class = feature_channels // out_size
            self.fc.weight.data = torch.zeros_like(self.fc.weight.data)
            self.fc.bias.data = torch.zeros_like(self.fc.bias.data)
            for c in range(out_size):
                start = c * features_per_class
                end = (c+1) * features_per_class
                self.fc.weight.data[start: end, c] = torch.ones_like(self.fc.weight.data[start: end, c])

    def forward(self, x, visualize_dir=None):
        if visualize_dir is not None:
            bs = min(32, x.shape[0])
            for j in range(bs):
                img_path = os.path.join(visualize_dir, str(j), "input.png")
                os.makedirs(os.path.dirname(img_path))
                ts.save(x[j], img_path)
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if visualize_dir is not None:
                for j in range(bs):
                    layer_path = os.path.join(visualize_dir, str(j), f"layer_{i}_features.png")
                    ts.save(x[j], layer_path)
        flat = x.view(x.shape[0], -1)
        fc_out = self.fc(flat)
        return fc_out

    def trainable_parameters(self):
        if self.mod_last_layer is not None:
            return self.conv_layers.parameters()
        else:
            return self.parameters()
