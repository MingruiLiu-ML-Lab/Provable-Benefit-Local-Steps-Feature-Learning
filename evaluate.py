"""
Evaluating the model on the test set.
"""

import torch

from data_loader import MultiClientLoader


def evaluate(test_loader, net, criterion, dataset_name):
    """
    Args:
        test_loader: an iterator over the test set.
        net: the neural network model employed.
        criterion: the loss function.

    Outputs:
        Average loss and accuracy achieved by the model in the test set.
    """
    net.eval()

    loss = 0.0
    accuracy = 0
    total = 0
    with torch.no_grad():

        # Hacky way to evaluate Torch DataLoaders and MultiClientLoaders.
        if isinstance(test_loader, MultiClientLoader):
            while True:
                try:
                    data = test_loader.next(cycle=False)
                    l, a, t = evaluate_batch(data, net, criterion, dataset_name)
                    loss += l
                    accuracy += a
                    total += t
                except StopIteration:
                    break
        else:
            for data in test_loader:
                l, a, t = evaluate_batch(data, net, criterion, dataset_name)
                loss += l
                accuracy += a
                total += t

        loss = loss.item() / total
        accuracy = accuracy / total

    return (loss, accuracy)


def evaluate_batch(data, net, criterion, dataset_name):
    inputs, labels = data
    labels = labels.cuda()
    outputs = predict(net, inputs, dataset_name)
    loss = criterion(outputs, labels) * labels.size(0)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == labels).sum().item()
    total = labels.size(0)
    return loss, accuracy, total


def predict(net, inputs, dataset_name):
    """ Get predictions for a single batch. """
    if dataset_name == "SNLI":
        (s1_embed, s2_embed), (s1_lens, s2_lens) = inputs
        s1_embed, s2_embed = s1_embed.cuda(), s2_embed.cuda()
        outputs = net((s1_embed, s1_lens), (s2_embed, s2_lens))
    elif dataset_name == "Sent140":
        sent_embed, sent_lens = inputs
        sent_embed = sent_embed.cuda()
        outputs = net((sent_embed, sent_lens))
    else:
        inputs = inputs.cuda()
        outputs = net(inputs)
    return outputs
