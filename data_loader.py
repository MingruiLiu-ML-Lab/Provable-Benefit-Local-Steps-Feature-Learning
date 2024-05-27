"""
Create train, valid, test iterators for a chosen dataset.
"""

import os
import random
import pickle
import json
import glob
import csv
import PIL
import copy
from math import ceil
from collections import namedtuple
from typing import List, Union, Optional, Tuple, Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from test import test_dataset


GLOVE_NAME = "glove.840B.300d.txt"
GLOVE_DIM = 300
VOCAB_NAME = "vocab.pkl"
WORDVEC_NAME = "wordvec.pkl"

DEBUG = False

CSV = namedtuple("CSV", ["header", "index", "data"])


class MultiClientLoader:
    """
    Data loader for multiple client datasets, but only loads data from one client at a
    time.
    """

    def __init__(
        self,
        train_set: torch.utils.data.Dataset,
        client_subidxs: List[List[int]],
        batch_size: int,
        current_clients: List[int] = None,
        **kwargs,
    ) -> None:
        """
        Arguments
        ---------
        train_set: torch dataset holding combined dataset of all clients.
        client_subidxs (List[List[int]]): A list whose the i-th element is a list of
            idxs of the members of `train_set` belonging to the local dataset of client
            i.
        current_clients (List[int]): Client(s) whose dataset to load from.

        Extra arguments are passed to the constructor of DataLoader whenever it is
        created for a new client.
        """

        # Store state.
        self.train_set = train_set
        self.client_subidxs = client_subidxs
        self.batch_size = batch_size
        self.num_clients = len(self.client_subidxs)
        self.dataloader_kwargs = dict(kwargs)

        # Set current client, if necessary.
        self.current_clients = None
        self._current_iter = None
        if current_clients is not None:
            self.set_clients(current_clients)

    def set_clients(self, current_clients: List[int]) -> None:
        """
        Set the current client(s). Note that we only construct a new DataLoader if the
        set value of current_clients differs from the previous value, to avoid the
        overhead of constructing DataLoader if possible.
        """
        assert current_clients is not None
        if self.current_clients is None:
            prev_clients = None
        else:
            prev_clients = set(self.current_clients)
        self.current_clients = list(current_clients)
        current_idxs = []
        for c in self.current_clients:
            current_idxs += self.client_subidxs[c]
        if prev_clients != set(self.current_clients):
            self.current_loader = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                sampler=SubsetRandomSampler(current_idxs),
                **self.dataloader_kwargs,
            )
        self.reset()

    def next(self, cycle=True) -> None:
        """ Load the next batch from the current client. """

        if self.current_clients is None:
            raise ValueError("Can't load data until `current_clients` is set.")

        try:
            return next(self._current_iter)
        except StopIteration:
            if cycle:
                self.reset()
                return next(self._current_iter)
            else:
                raise

    def reset(self) -> None:
        """ Resets iteration over the current client dataset. """

        if self.current_clients is None:
            raise ValueError(
                "Can't reset MultiClientLoader until `current_clients` is set."
            )
        self._current_iter = iter(self.current_loader)


def data_loader(
    dataset_name,
    dataroot,
    batch_size,
    val_ratio,
    total_clients,
    world_size,
    rank,
    group,
    heterogeneity=0,
    extra_bs=None,
    num_workers=1,
    small=False,
    binarize_classes=False,
    no_data_augment=False,
    feature_noise=0.0,
):
    """
    Args:
        dataset_name (str): the name of the dataset to use, currently only
            supports 'MNIST', 'FashionMNIST', 'CIFAR10' and 'CIFAR100'.
        dataroot (str): the location to save the dataset.
        batch_size (int): batch size used in training.
        val_ratio (float): the percentage of trainng data used as validation, if there
            is no separate validation set.
        total_clients (int): total number of clients participating in training.
        world_size (int): how many processes will be used in training.
        rank (int): the rank of this process.
        heterogeneity (float): dissimilarity between data distribution across clients.
            Between 0 and 1.
        extra_bs (int): Batch size for extra data loader.
        small (bool): Whether to use miniature dataset.

    Outputs:
        iterators over training, validation, and test data.
    """
    if ((val_ratio < 0) or (val_ratio > 1.0)):
        raise ValueError("[!] val_ratio should be in the range [0, 1].")
    if heterogeneity < 0:
        raise ValueError("Data heterogeneity must be positive.")
    if total_clients == 1 and heterogeneity > 0:
        raise ValueError("Cannot create a heterogeneous dataset when total_clients == 1.")

    if no_data_augment and dataset_name != "FeatureCIFAR":
        raise NotImplementedError

    # Dataset-specific processing.
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              normalize])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             normalize])
        train_kwargs = {"train": True, "download": True, "transform": transform_train}
        val_kwargs = {"train": True, "download": True, "transform": transform_test}
        test_kwargs = {"train": False, "download": True, "transform": transform_test}
        num_labels = 10
        separate_val = False
        predefined_clients = False

    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize((0.5071, 0.4866, 0.4409),
                                         (0.2673, 0.2564, 0.2762))
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              normalize])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             normalize])
        train_kwargs = {"train": True, "download": True, "transform": transform_train}
        val_kwargs = {"train": True, "download": True, "transform": transform_test}
        test_kwargs = {"train": False, "download": True, "transform": transform_test}
        num_labels = 100
        separate_val = False
        predefined_clients = False

    elif dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              normalize])
        transform_test = transform_train
        train_kwargs = {"train": True, "download": True, "transform": transform_train}
        val_kwargs = {"train": True, "download": True, "transform": transform_test}
        test_kwargs = {"train": False, "download": True, "transform": transform_test}
        num_labels = 10
        separate_val = False
        predefined_clients = False

    elif dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST
        normalize = transforms.Normalize((0.2860,), (0.3530,))
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              normalize])
        transform_test = transform_train
        train_kwargs = {"train": True, "download": True, "transform": transform_train}
        val_kwargs = {"train": True, "download": True, "transform": transform_test}
        test_kwargs = {"train": False, "download": True, "transform": transform_test}
        num_labels = 10
        separate_val = False
        predefined_clients = False

    elif dataset_name == 'FEMNIST':
        dataset = FEMNISTDataset
        train_kwargs = {"split": "train"}
        val_kwargs = {"split": "train"}
        test_kwargs = {"split": "test"}
        num_labels = 62
        separate_val = False
        predefined_clients = True

    elif dataset_name == "SNLI":
        dataset = SNLIDataset
        train_kwargs = {"split": "train"}
        val_kwargs = {"split": "dev"}
        test_kwargs = {"split": "test"}
        num_labels = 3
        separate_val = True
        predefined_clients = False

    elif dataset_name == 'Sent140':
        dataset = Sent140Dataset
        train_kwargs = {"split": "train"}
        val_kwargs = {"split": "train"}
        test_kwargs = {"split": "test"}
        num_labels = 2
        separate_val = False
        predefined_clients = True

    elif dataset_name == "CelebA":
        dataset = CelebA
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])
        train_kwargs = {"split": "train", "transform": transform}
        val_kwargs = {"split": "valid", "transform": transform}
        test_kwargs = {"split": "test", "transform": transform}
        num_labels = 2
        separate_val = True
        predefined_clients = False

    elif dataset_name == 'FeatureCIFAR':
        assert binarize_classes
        dataset = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))
        if no_data_augment:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  normalize])
        else:
            transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.ToTensor(),
                                                  normalize])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             normalize])

        train_kwargs = {"train": True, "download": True, "transform": transform_train}
        val_kwargs = {"train": True, "download": True, "transform": transform_test}
        test_kwargs = {"train": False, "download": True, "transform": transform_test}
        num_labels = 2
        separate_val = False
        predefined_clients = False

    else:
        raise NotImplementedError

    # load and split the train dataset into train and validation.
    train_set = dataset(root=dataroot, **train_kwargs)
    val_set = dataset(root=dataroot, **val_kwargs)
    test_set = dataset(root=dataroot, **test_kwargs)

    # Binarize classes.
    if binarize_classes:
        num_labels = 2
        if dataset_name in ["CIFAR10", "CIFAR100", "MNIST", "FeatureCIFAR"]:
            for dset in [train_set, val_set, test_set]:
                dset.original_targets = copy.deepcopy(dset.targets)
                for idx in range(len(dset.targets)):
                    dset.targets[idx] = dset.targets[idx] % 2
        else:
            raise NotImplementedError

    # Partition to create feature heterogeneity for CelebA.
    if dataset_name in ["CelebA", "FeatureCIFAR"]:
        loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}

        separate_minibatch = False

        if dataset_name == "CelebA":
            assert separate_val
            client_train_idxs = train_set.heterogeneity_partition(total_clients, heterogeneity)
            val_idxs = list(range(len(val_set)))

        elif dataset_name == "FeatureCIFAR":
            assert not separate_val
            combined_idxs = list(range(len(train_set)))
            random.shuffle(combined_idxs)
            val_split = round(len(train_set) * val_ratio)
            val_idxs = combined_idxs[:val_split]
            train_idxs = combined_idxs[val_split:]

            # Separate client datasets (even for Minibatch SGD) to create feature noise.
            separate_minibatch = feature_noise > 0 and total_clients == 1
            if separate_minibatch:
                original_clients = 1
                total_clients = 8

            client_train_idxs = feature_heterogeneity_partition(train_set, train_idxs, heterogeneity, num_labels, total_clients)

            # Separate validation and test datasets to create same feature noise in evaluation data.
            client_val_idxs = feature_heterogeneity_partition(train_set, val_idxs, heterogeneity, num_labels, total_clients)
            client_test_idxs = feature_heterogeneity_partition(test_set, list(range(len(test_set))), heterogeneity, num_labels, total_clients)

        else:
            raise NotImplementedError

        # Add feature noise for feature learning experiments.
        if feature_noise > 0:
            assert dataset_name == "FeatureCIFAR"
            for i in range(total_clients):
                other_clients = [k for k in range(total_clients) if k != i]
                j = random.choice(other_clients)
                client_noise_idx = random.choice(client_train_idxs[j])
                client_noise = train_set.data[client_noise_idx]

                dsets = [train_set, train_set, test_set]
                idx_sets = [client_train_idxs, client_val_idxs, client_test_idxs]
                for k, (dset, idx_set) in enumerate(zip(dsets, idx_sets)):
                    current_idxs = np.array(idx_set[i], dtype="long")
                    dset.data[current_idxs] = dset.data[current_idxs] - feature_noise * client_noise

            # Recombine client datasets for Minibatch SGD.
            if separate_minibatch:
                total_clients = original_clients

                combined_train_idxs = []
                for i in range(len(client_train_idxs)):
                    combined_train_idxs += client_train_idxs[i]
                client_train_idxs = [combined_train_idxs]

                local_val_idxs = []
                for i in range(len(client_val_idxs)):
                    local_val_idxs += client_val_idxs[i]

                local_test_idxs = []
                for i in range(len(client_test_idxs)):
                    local_test_idxs += client_test_idxs[i]

        # Use regular combined validation and test datasets if did not separate
        # minibatch datasets.
        if not separate_minibatch:
            val_start = round(rank / total_clients * len(val_idxs))
            val_end = round((rank + 1) / total_clients * len(val_idxs))
            test_start = round(rank / total_clients * len(test_set))
            test_end = round((rank + 1) / total_clients * len(test_set))
            local_val_idxs = val_idxs[val_start: val_end]
            local_test_idxs = list(range(test_start, test_end))

        train_loader = MultiClientLoader(
            train_set, client_train_idxs, **loader_kwargs
        )
        val_loader = DataLoader(
            val_set, sampler=SubsetRandomSampler(local_val_idxs), **loader_kwargs
        )
        test_loader = DataLoader(
            test_set, sampler=SubsetRandomSampler(local_test_idxs), **loader_kwargs
        )
        assert extra_bs is None

        # Test dataset partitioning.
        if DEBUG:
            test_dataset(
                dataset_name,
                train_loader,
                client_train_idxs,
                local_val_idxs,
                local_test_idxs,
                separate_val,
                predefined_clients,
                num_labels,
                rank,
                group,
            )

        return train_loader, val_loader, test_loader, None

    # Handle the case of predefined clients (for datasets designed for federated
    # learning) versus non-predefined clients (for centralized datasets which we split
    # into clients for federated learning).
    if predefined_clients:
        get_idxs = lambda dset: list(range(dset.num_clients))
    else:
        get_idxs = lambda dset: list(range(len(dset)))

    # Split training set into training and validation.
    if separate_val:
        train_idxs = get_idxs(train_set)
        val_idxs = get_idxs(val_set)
        random.shuffle(val_idxs)
    else:
        total_train_idxs = get_idxs(train_set)
        random.shuffle(total_train_idxs)
        val_split = round(val_ratio * len(total_train_idxs))
        train_idxs = total_train_idxs[val_split:]
        val_idxs = total_train_idxs[:val_split]
    test_idxs = get_idxs(test_set)
    random.shuffle(test_idxs)

    # Partition the training data into multiple clients if needed. Data partitioning to
    # create heterogeneity is performed according to the specifications in
    # https://arxiv.org/abs/1910.06378.
    if total_clients > 1:
        random.seed(1234)

    # Split data into iid pool and non-iid pool. If we don't have predefined clients,
    # ensure that label distribution of iid pool and non-iid pool matches that of
    # overall dataset.
    if predefined_clients:

        # Client partitioning only works for binary classification, since we sort
        # clients by proportion of positive samples.
        if train_set.n_classes != 2 and heterogeneity > 0:
            raise NotImplementedError

        # Divide clients into iid pool and non-iid pool.
        random.shuffle(train_idxs)
        iid_split = round((1.0 - heterogeneity) * len(train_idxs))
        iid_pool = train_idxs[:iid_split]
        non_iid_pool = train_idxs[iid_split:]

        # Sort non-iid pool by proportion of positive samples.
        positive_prop = np.zeros(len(non_iid_pool))
        for i, client in enumerate(non_iid_pool):
            client_labels = [train_set.labels[idx] for idx in train_set.user_items[client]]
            positive_prop[i] = np.mean(client_labels)
        pool_order = np.argsort(positive_prop)
        non_iid_pool = [non_iid_pool[idx] for idx in pool_order]

    else:

        # Collect indices of instances with each label.
        train_label_idxs = get_label_indices(dataset_name, train_set, num_labels)
        for l in range(num_labels):
            if not separate_val:
                train_label_idxs[l] = [i for i in train_label_idxs[l] if i in train_idxs]
            random.shuffle(train_label_idxs[l])

        # Divide samples from each label into iid pool and non-iid pool. Note that samples
        # in iid pool are shuffled while samples in non-iid pool are sorted by label.
        iid_pool = []
        non_iid_pool = []
        for i in range(num_labels):
            iid_split = round((1.0 - heterogeneity) * len(train_label_idxs[i]))
            iid_pool += train_label_idxs[i][:iid_split]
            non_iid_pool += train_label_idxs[i][iid_split:]
        random.shuffle(iid_pool)

    # Allocate iid and non-iid samples to each client.
    client_train_idxs = [[] for _ in range(total_clients)]
    num_iid = len(iid_pool) // total_clients
    num_non_iid = len(non_iid_pool) // total_clients
    partition_size = num_iid + num_non_iid
    for j in range(total_clients):
        client_train_idxs[j] += iid_pool[num_iid * j: num_iid * (j+1)]
        client_train_idxs[j] += non_iid_pool[num_non_iid * j: num_non_iid * (j+1)]
        random.shuffle(client_train_idxs[j])

    # Get indices of local validation and test dataset. Note that the validation and
    # test set are not split into `total_clients` partitions, just `world_size`
    # partitions, since evaluation does not need to happen separate for each client.
    # TODO: Do we really need to enforce that each partition of the test set has the
    # same size? We are just throwing away some test examples here and it may be
    # unnecessary.
    val_partition = len(val_idxs) // world_size
    test_partition = len(test_idxs) // world_size
    local_val_idxs = val_idxs[rank * val_partition: (rank+1) * val_partition]
    local_test_idxs = test_idxs[rank * test_partition: (rank+1) * test_partition]

    # Use miniature dataset, if necessary.
    if small:
        for r in range(total_clients):
            client_train_idxs[r] = client_train_idxs[r][:round(len(client_train_idxs[r]) / 100)]
        local_val_idxs = local_val_idxs[:round(len(local_val_idxs) / 100)]
        local_test_idxs = local_test_idxs[:round(len(local_test_idxs) / 100)]

    # If partitioning clients, convert list of clients into list of dataset elements.
    if predefined_clients:

        for r in range(total_clients):
            current_idxs = list(client_train_idxs[r])
            client_train_idxs[r] = []
            for i in current_idxs:
                client_train_idxs[r] += train_set.user_items[i]

        current_idxs = list(local_val_idxs)
        local_val_idxs = []
        for i in current_idxs:
            local_val_idxs += val_set.user_items[i]

        current_idxs = list(local_test_idxs)
        local_test_idxs = []
        for i in current_idxs:
            local_test_idxs += test_set.user_items[i]

    # Construct loaders for train, val, test, and extra sets. To support client
    # subsampling, we use a MultiClientLoader for the training loader, which allows each
    # worker process to switch between client datasets at the beginning of each round.
    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    if dataset_name == "SNLI":
        loader_kwargs["collate_fn"] = collate_pad_double
    elif dataset_name == "Sent140":
        loader_kwargs["collate_fn"] = collate_pad
    train_loader = MultiClientLoader(
        train_set, client_train_idxs, **loader_kwargs
    )
    val_loader = DataLoader(
        val_set, sampler=SubsetRandomSampler(local_val_idxs), **loader_kwargs
    )
    test_loader = DataLoader(
        test_set, sampler=SubsetRandomSampler(local_test_idxs), **loader_kwargs
    )
    extra_loader = None
    if extra_bs is not None:
        extra_loader = MultiClientLoader(
            train_set, client_train_idxs, **loader_kwargs
        )

    # Test dataset partitioning.
    if DEBUG:
        test_dataset(
            dataset_name,
            train_loader,
            client_train_idxs,
            local_val_idxs,
            local_test_idxs,
            separate_val,
            predefined_clients,
            num_labels,
            rank,
            group,
        )

    return train_loader, val_loader, test_loader, extra_loader


class SNLIDataset(torch.utils.data.Dataset):

    def __init__(self, root="", split="train"):
        """ Initialize SNLI dataset. """

        assert split in ["train", "dev", "test"]
        self.root = os.path.join(root, "snli_1.0")
        self.split = split
        self.embed_dim = GLOVE_DIM
        self.n_classes = 3

        """ Read and store data from files. """
        self.labels = ["entailment", "neutral", "contradiction"]
        labels_to_idx = {label: i for i, label in enumerate(self.labels)}

        # Read sentence and label data for current split from files.
        s1_path = os.path.join(self.root, "SNLI", f"s1.{self.split}")
        s2_path = os.path.join(self.root, "SNLI", f"s2.{self.split}")
        target_path = os.path.join(self.root, "SNLI", f"labels.{self.split}")
        self.s1_sentences = [line.rstrip() for line in open(s1_path, "r")]
        self.s2_sentences = [line.rstrip() for line in open(s2_path, "r")]
        self.targets = np.array(
            [labels_to_idx[line.rstrip("\n")] for line in open(target_path, "r")]
        )
        assert len(self.s1_sentences) == len(self.s2_sentences)
        assert len(self.s1_sentences) == len(self.targets)
        self.dataset_size = len(self.s1_sentences)
        print(f"Loaded {self.dataset_size} sentence pairs for {self.split} split.")

        # If vocab exists on file, load it. Otherwise, read sentence data for all splits
        # from files to build vocab.
        vocab_path = os.path.join(self.root, "SNLI", VOCAB_NAME)
        if os.path.isfile(vocab_path):
            print("Loading vocab.")
            with open(vocab_path, "rb") as vocab_file:
                vocab = pickle.load(vocab_file)
        else:
            print(
                "Constructing vocab. This only needs to be done once but will take "
                "several minutes."
            )
            vocab = ["<s>", "</s>"]
            for split in ["train", "dev", "test"]:
                paths = [
                    os.path.join(self.root, "SNLI", f"s1.{split}"),
                    os.path.join(self.root, "SNLI", f"s2.{split}"),
                ]
                for path in paths:
                    for line in open(path, "r"):
                        for word in line.rstrip().split():
                            if word not in vocab:
                                vocab.append(word)
            with open(vocab_path, "wb") as vocab_file:
                pickle.dump(vocab, vocab_file)
        print(f"Loaded vocab with {len(vocab)} words.")

        # Read in GLOVE vectors and store mapping from words to vectors.
        self.word_vec = {}
        glove_path = os.path.join(self.root, "GloVe", GLOVE_NAME)
        wordvec_path = os.path.join(self.root, "SNLI", WORDVEC_NAME)
        if os.path.isfile(wordvec_path):
            print("Loading word vector mapping.")
            with open(wordvec_path, "rb") as wordvec_file:
                self.word_vec = pickle.load(wordvec_file)
        else:
            print(
                "Constructing mapping from vocab to word vectors. This only needs to "
                "be done once but can take up to 30 minutes."
            )
            with open(glove_path, "r") as glove_file:
                for line in glove_file:
                    word, vec = line.split(' ', 1)
                    if word in vocab:
                        self.word_vec[word] = np.array(list(map(float, vec.split())))
            with open(wordvec_path, "wb") as wordvec_file:
                pickle.dump(self.word_vec, wordvec_file)
        print(f"Found {len(self.word_vec)}/{len(vocab)} words with glove vectors.")

        # Split each sentence into words, add start/stop tokens to the beginning/end of
        # each sentence, and remove any words which do not have glove embeddings.
        assert "<s>" in vocab
        assert "</s>" in vocab
        assert "<s>" in self.word_vec
        assert "</s>" in self.word_vec
        for i in range(len(self.s1_sentences)):
            sent = self.s1_sentences[i]
            self.s1_sentences[i] = np.array(
                ["<s>"] +
                [word for word in sent.split() if word in self.word_vec] +
                ["</s>"]
            )
        for i in range(len(self.s2_sentences)):
            sent = self.s2_sentences[i]
            self.s2_sentences[i] = np.array(
                ["<s>"] +
                [word for word in sent.split() if word in self.word_vec] +
                ["</s>"]
            )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """ Return a single element of the dataset. """

        # Encode sentences as sequence of glove vectors.
        sent1 = self.s1_sentences[idx]
        sent2 = self.s2_sentences[idx]
        s1_embed = np.zeros((len(sent1), GLOVE_DIM))
        s2_embed = np.zeros((len(sent2), GLOVE_DIM))
        for j in range(len(sent1)):
            s1_embed[j] = self.word_vec[sent1[j]]
        for j in range(len(sent2)):
            s2_embed[j] = self.word_vec[sent2[j]]
        s1_embed = torch.from_numpy(s1_embed).float()
        s2_embed = torch.from_numpy(s2_embed).float()

        # Convert targets to tensor.
        target = torch.tensor([self.targets[idx]]).long()

        return (s1_embed, s2_embed), target

    @property
    def n_words(self):
        return len(self.word_vec)


class Sent140Dataset(torch.utils.data.Dataset):

    def __init__(self, root="", split="train"):
        """ Initialize Sent140 dataset. """

        assert split in ["train", "test"]
        self.root = os.path.join(root, "sent140")
        self.data_path = os.path.join(self.root, f"{split}.json")
        self.split = split
        self.embed_dim = GLOVE_DIM
        self.n_classes = 2

        # Read sentence and label data for current split from file.
        with open(self.data_path, "r") as f:
            all_data = json.load(f)
        self.users = range(len(all_data["users"]))
        self.num_clients = len(self.users)
        self.sentences = []
        self.labels = []
        self.user_items = {}
        def process_label(l):
            if l == "0":
                return 0
            elif l == "4":
                return 1
            else:
                raise ValueError

        j = 0
        for i in self.users:
            user = all_data["users"][i]
            self.user_items[i] = []
            tweets = all_data["user_data"][user]["x"]
            labels = all_data["user_data"][user]["y"]
            assert len(tweets) == len(labels)
            for tweet_data, label in zip(tweets, labels):
                self.sentences.append(tweet_data[4])
                self.labels.append(process_label(label))
                self.user_items[i].append(j)
                j += 1

        # If vocab exists on file, load it. Otherwise, read sentence data for all splits
        # from files to build vocab.
        vocab_path = os.path.join(self.root, VOCAB_NAME)
        if os.path.isfile(vocab_path):
            print("Loading vocab.")
            with open(vocab_path, "rb") as vocab_file:
                vocab = pickle.load(vocab_file)
        else:
            print(
                "Constructing vocab. This only needs to be done once but will take "
                "several minutes."
            )
            vocab = ["<s>", "</s>"]
            for split in ["train", "test"]:
                path = os.path.join(self.root, f"{split}.json")
                with open(path, "r") as f:
                    split_data = json.load(f)
                split_sentences = split_data["user_data"]

                for user in tqdm(split_data["users"]):
                    for tweet_data in split_data["user_data"][user]["x"]:
                        sentence = tweet_data[4]
                        for word in sentence.rstrip().split():
                            if word not in vocab:
                                vocab.append(word)
            with open(vocab_path, "wb") as vocab_file:
                pickle.dump(vocab, vocab_file)
        print(f"Loaded vocab with {len(vocab)} words.")

        # Read in GLOVE vectors and store mapping from words to vectors.
        self.word_vec = {}
        glove_path = os.path.join(self.root, GLOVE_NAME)
        wordvec_path = os.path.join(self.root, WORDVEC_NAME)
        if os.path.isfile(wordvec_path):
            print("Loading word vector mapping.")
            with open(wordvec_path, "rb") as wordvec_file:
                self.word_vec = pickle.load(wordvec_file)
        else:
            print(
                "Constructing mapping from vocab to word vectors. This only needs to "
                "be done once but can take up to 30 minutes."
            )
            lines = []
            with open(glove_path, "r") as glove_file:
                for line in glove_file:
                    lines.append(line)
            for line in tqdm(lines):
                word, vec = line.split(' ', 1)
                if word in vocab:
                    self.word_vec[word] = np.array(list(map(float, vec.split())))
            with open(wordvec_path, "wb") as wordvec_file:
                pickle.dump(self.word_vec, wordvec_file)
        print(f"Found {len(self.word_vec)}/{len(vocab)} words with glove vectors.")

        # Split each sentence into words, add start/stop tokens to the beginning/end of
        # each sentence, and remove any words which do not have glove embeddings.
        assert "<s>" in vocab
        assert "</s>" in vocab
        assert "<s>" in self.word_vec
        assert "</s>" in self.word_vec
        for i, sentence in enumerate(self.sentences):
            self.sentences[i] = np.array(
                ["<s>"] +
                [word for word in sentence.split() if word in self.word_vec] +
                ["</s>"]
            )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """ Return a single element of the dataset. """

        # Encode sentence as sequence of glove vectors.
        sent = self.sentences[idx]
        sent_embed = np.zeros((len(sent), GLOVE_DIM))
        for j in range(len(sent)):
            sent_embed[j] = self.word_vec[sent[j]]
        sent_embed = torch.from_numpy(sent_embed).float()

        # Convert label to tensor.
        target = torch.tensor([self.labels[idx]]).long()

        return sent_embed, target

    @property
    def n_words(self):
        return len(self.word_vec)


class FEMNISTDataset(torch.utils.data.Dataset):

    def __init__(self, root="", split="train"):
        assert split in ["train", "test"]
        self.split = split
        self.root = os.path.join(root, "femnist", self.split)
        self.n_classes = 62
        self.img_shape = (1, 28, 28)

        # Read and store data from files.
        self.users = []
        self.imgs = []
        self.targets = []
        self.user_items = {}
        j = 0
        data_files = glob.glob(os.path.join(self.root, "*.json"))
        for data_file in data_files:
            with open(data_file, "r") as data_f:
                current_data = json.load(data_f)
            for user in current_data["users"]:
                if user not in self.users:
                    user_idx = len(self.users)
                    self.users.append(user)
                    self.user_items[user_idx] = []
                else:
                    user_idx = self.users.index(user)
                user_x = current_data["user_data"][user]["x"]
                user_y = current_data["user_data"][user]["y"]
                for x, y in zip(user_x, user_y):
                    self.imgs.append(
                        torch.FloatTensor(x).reshape(self.img_shape)
                    )
                    self.targets.append(int(y))
                    self.user_items[user_idx].append(j)
                    j += 1

        self.imgs = torch.stack(self.imgs)
        self.targets = torch.LongTensor(self.targets)
        self.num_clients = len(self.users)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.targets[idx]


class CelebA(torch.utils.data.Dataset):
    """ CelebA dataset, modified from Pytorch.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    target_attr = 20 # predict male vs. female
    attr_priority = [
        20, 36, 18, 24, 0, 1, 2, 34, 7, 33, 38, 9, 16, 3, 30, 39, 37, 19, 22, 12, 13,
        29, 27, 14, 15, 6, 4, 17, 5, 31, 35, 8, 28, 25, 11, 21, 26, 32, 10, 23
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_idx = split_map[split]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_idx is None else (splits.data == split_idx).squeeze()
        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]

        self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, self.target_attr])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def heterogeneity_partition(self, total_clients, heterogeneity):

        # Break into iid pool and noniid pool.
        dset_size = len(self.attr)
        sample_idxs = torch.randperm(dset_size)
        het_boundary = round(heterogeneity * dset_size)
        noniid_pool = sample_idxs[:het_boundary]
        iid_pool = sample_idxs[het_boundary:]

        # Partition noniid pool by target.
        noniid_off_pos = (self.attr[noniid_pool, self.target_attr] == 0).nonzero().squeeze(1)
        noniid_on_pos = (self.attr[noniid_pool, self.target_attr] != 0).nonzero().squeeze(1)
        noniid_off_pool = noniid_pool[noniid_off_pos]
        noniid_on_pool = noniid_pool[noniid_on_pos]

        # Sort noniid pools according to attr_priority.
        noniid_off_labels = [(idx, label.tolist()) for (idx, label) in zip(noniid_off_pool, self.attr[noniid_off_pool])]
        noniid_on_labels = [(idx, label.tolist()) for (idx, label) in zip(noniid_on_pool, self.attr[noniid_on_pool])]
        for feature_idx in range(len(self.attr_priority)-1, -1, -1):
            noniid_off_labels = sorted(noniid_off_labels, key=lambda x: x[1][feature_idx])
            noniid_on_labels = sorted(noniid_on_labels, key=lambda x: x[1][feature_idx])
        noniid_off_pool = [idx for (idx, label) in noniid_off_labels]
        noniid_on_pool = [idx for (idx, label) in noniid_on_labels]

        # Allocate iid and noniid data to each client.
        client_train_idxs = [[] for _ in range(total_clients)]
        iid_partition = len(iid_pool) // total_clients
        noniid_off_partition = len(noniid_off_pool) // total_clients
        noniid_on_partition = len(noniid_on_pool) // total_clients
        for client in range(total_clients):
            client_train_idxs[client] += iid_pool[client * iid_partition: (client+1) * iid_partition]
            client_train_idxs[client] += noniid_off_pool[client * noniid_off_partition: (client+1) * noniid_off_partition]
            client_train_idxs[client] += noniid_on_pool[client * noniid_on_partition: (client+1) * noniid_on_partition]

        return client_train_idxs


def feature_heterogeneity_partition(train_set, train_idxs, heterogeneity, num_labels, total_clients):

    # Break train set into iid pool and noniid pool.
    dset_size = len(train_idxs)
    sample_idxs = torch.randperm(dset_size)
    het_boundary = round(heterogeneity * dset_size)
    noniid_pool = sample_idxs[:het_boundary]
    iid_pool = sample_idxs[het_boundary:]

    # Partition noniid pool by target.
    noniid_pools = [[] for _ in range(num_labels)]
    for idx in noniid_pool:
        label = train_set.targets[idx]
        noniid_pools[label].append(idx)

    # Sort noniid pools according to original label.
    for pool in range(num_labels):
        noniid_digits = [(idx, train_set.original_targets[idx]) for idx in noniid_pools[pool]]
        noniid_digits = sorted(noniid_digits, key=lambda x: x[1])
        noniid_pools[pool] = [idx for (idx, _) in noniid_digits]

    # Allocate iid and noniid data to each client.
    client_train_idxs = [[] for _ in range(total_clients)]
    iid_partition = len(iid_pool) // total_clients
    noniid_partitions = [len(pool) // total_clients for pool in noniid_pools]
    for client in range(total_clients):
        client_train_idxs[client] += iid_pool[client * iid_partition: (client+1) * iid_partition]
        for pool in range(num_labels):
            start = client * noniid_partitions[pool]
            end = (client + 1) * noniid_partitions[pool]
            client_train_idxs[client] += noniid_pools[pool][start: end]

    return client_train_idxs


def collate_pad_double(data_points):
    """ Pad data points with zeros to fit length of longest data point in batch. """

    s1_embeds = [x[0][0] for x in data_points]
    s2_embeds = [x[0][1] for x in data_points]
    targets = [x[1] for x in data_points]

    # Get sentences for batch and their lengths.
    s1_lens = np.array([sent.shape[0] for sent in s1_embeds])
    max_s1_len = np.max(s1_lens)
    s2_lens = np.array([sent.shape[0] for sent in s2_embeds])
    max_s2_len = np.max(s2_lens)
    lens = (s1_lens, s2_lens)

    # Encode sentences as glove vectors.
    bs = len(data_points)
    s1_embed = np.zeros((max_s1_len, bs, GLOVE_DIM))
    s2_embed = np.zeros((max_s2_len, bs, GLOVE_DIM))
    for i in range(bs):
        e1 = s1_embeds[i]
        e2 = s2_embeds[i]
        s1_embed[: len(e1), i] = e1.clone()
        s2_embed[: len(e2), i] = e2.clone()
    embeds = (
        torch.from_numpy(s1_embed).float(), torch.from_numpy(s2_embed).float()
    )

    # Convert targets to tensor.
    targets = torch.cat(targets)

    return (embeds, lens), targets


def collate_pad(data_points):
    """ Pad data points with zeros to fit length of longest data point in batch. """

    sent_embeds = [x[0] for x in data_points]
    targets = [x[1] for x in data_points]

    # Get sentences for batch and their lengths.
    lens = np.array([sent.shape[0] for sent in sent_embeds])
    max_sent_len = np.max(lens)

    # Encode sentences as glove vectors.
    bs = len(data_points)
    sent_embed = np.zeros((max_sent_len, bs, GLOVE_DIM))
    for i in range(bs):
        e = sent_embeds[i]
        sent_embed[: len(e), i] = e.clone()
    sent_embed = torch.from_numpy(sent_embed).float()

    # Convert targets to tensor.
    targets = torch.cat(targets)

    return (sent_embed, lens), targets


def get_label_indices(dataset_name, dset, num_labels):
    """
    Returns a dictionary mapping each label to a list of the indices of elements in
    `dset` with the corresponding label.
    """

    if dataset_name in ["CIFAR10", "CIFAR100", "MNIST"]:
        label_indices = [[] for _ in range(num_labels)]
        for idx, label in enumerate(dset.targets):
            label_indices[label].append(idx)
    elif dataset_name == "SNLI":
        label_indices = [
            (dset.targets == i).nonzero()[0].tolist()
            for i in range(dset.n_classes)
        ]
    else:
        raise NotImplementedError

    return label_indices


def get_num_classes(dataset_name):
    if dataset_name == "CIFAR10":
        classes = 10
    else:
        raise NotImplementedError
    return classes
