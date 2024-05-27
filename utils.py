import numpy as np
import torch


def kl_div(P, Q):
    """
    Compute KL divergence between two categorical distributions. We clamp the operand of
    log in order to avoid numerical instability.
    """
    return (P * (torch.clamp(P / Q, min=1e-8)).log()).sum()


def debug(x, rank):
    with open(f"DEBUG_{rank}", "a+") as f:
        f.write(f"{rank}: {x}\n")


def get_client_label_dist(dataset_name, loader, num_classes):
    """
    Compute the label distribution for each client in a MultiClientLoader.
    """

    # Separate computation for CelebA.
    if dataset_name == "CelebA":
        client_label_dist = []
        for client in range(loader.num_clients):
            client_idxs = loader.client_subidxs[client]
            client_label_dist.append(
                torch.mean(loader.train_set.attr[client_idxs].float(), dim=0)
            )
        return client_label_dist

    # Iterate through data of each client and compute label distribution.
    client_label_dist = []
    for r in range(loader.num_clients):
        loader.set_clients([r])
        label_dist = torch.zeros(num_classes)
        while True:
            try:
                _, labels = loader.next(cycle=False)
            except StopIteration:
                break
            for c in range(num_classes):
                label_dist[c] += torch.sum(labels == c)
        label_dist = label_dist / torch.sum(label_dist)
        client_label_dist.append(label_dist.clone())

    client_label_dist = torch.stack(client_label_dist)

    # Reset iterator for each client.
    for r in range(loader.num_clients):
        loader.set_clients([r])

    # Test original labels for FeatureCIFAR.
    if dataset_name == "FeatureCIFAR":
        num_digits = 10
        client_digit_dist = []
        for client in range(loader.num_clients):
            client_idxs = loader.client_subidxs[client]
            client_digits = np.array([
                loader.train_set.original_targets[idx] for idx in client_idxs
            ])
            client_digit_dist.append(
                [np.sum(client_digits == c) / len(client_digits) for c in range(num_digits)]
            )
        return client_label_dist, client_digit_dist

    else:
        return client_label_dist


def trainable_parameters(net):
    if hasattr(net, "trainable_parameteters"):
        params = net.trainable_parameters()
    else:
        params = net.parameters()
    return params


class IdentityModule(torch.nn.Module):
    """ Dummy module to turn off BN in ResNet. """

    def __init__(self, *args, **kwargs):
        super(IdentityModule, self).__init__()

    def forward(self, x):
        return x
