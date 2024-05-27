import torch
import torch.distributed as dist

from utils import debug, get_client_label_dist


TOL = 1e-7


def test_corrections(
    local_corrections,
    global_correction,
    prev_local_corrections,
    round_avg_grad,
    round_clients,
    rank,
    group,
):
    """ Tests for SCAFFOLD corrections. """

    n_params = len(global_correction)
    total_clients = len(local_corrections)

    # Check that corrections have the right size.
    for i in range(len(local_corrections)):
        assert len(local_corrections[i]) == n_params

    # Test that everyone has the same local and global corrections.
    for i in range(total_clients):
        for j in range(n_params):
            min_correction = local_corrections[i][j].clone()
            max_correction = local_corrections[i][j].clone()
            dist.reduce(min_correction, dst=0, op=dist.ReduceOp.MIN, group=group)
            dist.reduce(max_correction, dst=0, op=dist.ReduceOp.MAX, group=group)
            if rank == 0:
                assert (min_correction == max_correction).all()
    for j in range(n_params):
        min_correction = global_correction[j].clone()
        max_correction = global_correction[j].clone()
        dist.reduce(min_correction, dst=0, op=dist.ReduceOp.MIN, group=group)
        dist.reduce(max_correction, dst=0, op=dist.ReduceOp.MAX, group=group)
        if rank == 0:
            assert (min_correction == max_correction).all()

    # Test that global correction is average of ALL local corrections.
    avg_local_correction = [0] * n_params
    for i in range(total_clients):
        for j in range(n_params):
            avg_local_correction[j] += local_corrections[i][j] / len(local_corrections)
    for j in range(n_params):
        compare_tensors(avg_local_correction[j], global_correction[j])

    # Test that local corrections are equal to average gradient over previous round for
    # sampled clients.
    clients_per_worker = len(round_avg_grad)
    for i in range(clients_per_worker):
        client = round_clients[rank * clients_per_worker + i]
        for j in range(n_params):
            compare_tensors(local_corrections[client][j], round_avg_grad[i][j])

    # Test that local corrections did not change for unsampled clients.
    for i in range(total_clients):
        if i not in round_clients:
            for j in range(n_params):
                compare_tensors(local_corrections[i][j], prev_local_corrections[i][j])

    print("SCAFFOLD tests passed!")


def test_initial_corrections(
    local_corrections,
    global_correction,
    rank,
    group,
):
    """ Tests for initial SCAFFOLD corrections. """

    n_params = len(global_correction)
    total_clients = len(local_corrections)

    # Check that corrections have the right size.
    for i in range(len(local_corrections)):
        assert len(local_corrections[i]) == n_params

    # Test that everyone has the same local and global corrections.
    for i in range(total_clients):
        for j in range(n_params):
            min_correction = local_corrections[i][j].clone()
            max_correction = local_corrections[i][j].clone()
            dist.reduce(min_correction, dst=0, op=dist.ReduceOp.MIN, group=group)
            dist.reduce(max_correction, dst=0, op=dist.ReduceOp.MAX, group=group)
            if rank == 0:
                assert (min_correction == max_correction).all()
    for j in range(n_params):
        min_correction = global_correction[j].clone()
        max_correction = global_correction[j].clone()
        dist.reduce(min_correction, dst=0, op=dist.ReduceOp.MIN, group=group)
        dist.reduce(max_correction, dst=0, op=dist.ReduceOp.MAX, group=group)
        if rank == 0:
            assert (min_correction == max_correction).all()

    # Test that global correction is average of ALL local corrections.
    avg_local_correction = [0] * n_params
    for i in range(total_clients):
        for j in range(n_params):
            avg_local_correction[j] += local_corrections[i][j] / len(local_corrections)
    for j in range(n_params):
        compare_tensors(avg_local_correction[j], global_correction[j])

    # Test that no local corrections are zero.
    for client in range(total_clients):
        for j in range(n_params):
            norm = torch.sum(local_corrections[client][j] ** 2)
            if float(norm) == 0.0:
                print(f"Rank, client, j, src: {args.rank}, {client}, {j}, {src}")
            assert norm != 0.0

    print("SCAFFOLD initial tests passed!")


def test_minibatch(net, rank, group):
    """ Tests for minibatch clip. """

    # Check that all devices have the same model, since we don't explicitly share the
    # model.
    for p in net.parameters():
        min_param = p.data.clone()
        max_param = p.data.clone()
        dist.reduce(min_param, dst=0, op=dist.ReduceOp.MIN, group=group)
        dist.reduce(max_param, dst=0, op=dist.ReduceOp.MAX, group=group)
        if rank == 0:
            assert (min_param == max_param).all()

    print("Minibatch tests passed!")


def test_dataset(
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
):

    print("Testing partitioned dataset...")
    total_clients = len(client_train_idxs)

    # Check that client training datasets were computed the same across workers.
    for i in range(total_clients):
        client_idxs = torch.tensor(client_train_idxs[i]).long().cuda()
        min_idxs = client_idxs.clone()
        max_idxs = client_idxs.clone()
        dist.reduce(min_idxs, dst=0, op=dist.ReduceOp.MIN, group=group)
        dist.reduce(max_idxs, dst=0, op=dist.ReduceOp.MAX, group=group)
        dist.broadcast(min_idxs, src=0, group=group)
        dist.broadcast(max_idxs, src=0, group=group)
        assert (min_idxs == max_idxs).all()

    # Check that each client training dataset is disjoint.
    sorted_client_train_idxs = []
    for r in range(total_clients):
        sorted_client_train_idxs.append(sorted(client_train_idxs[r]))
    dset_idx = 0
    client_dset_idxs = [0] * total_clients
    client_finished = [False] * total_clients
    while not all(client_finished) and dset_idx < len(train_loader.train_set):
        containing_clients = []
        for r in range(total_clients):
            if client_finished[r]:
                continue
            current_idx = client_dset_idxs[r]
            min_element = sorted_client_train_idxs[current_idx]
            if min_element == dset_idx:
                containing_clients.append(r)
                client_dset_idxs[r] += 1
                if client_dset_idxs[r] == len(sorted_client_train_idxs):
                    client_finished[r] = True
        assert len(containing_clients) <= 1
        dset_idx += 1

    # Check that training set and validation set are disjoint, if not using a separate
    # validation set.
    if not separate_val:
        for idx in local_val_idxs:
            for r in range(total_clients):
                assert idx not in client_train_idxs[r]

    # Print out label distribution and size for each client dataset.
    label_dist = get_client_label_dist(dataset_name, train_loader, num_labels)
    for r in range(total_clients):
        total = len(client_train_idxs[r])
        if dataset_name == "FeatureCIFAR":
            target_dist, digit_dist = label_dist
            print(f"Client {r} has {total} samples with ")
            print(f"    target distribution: {target_dist[r]}")
            print(f"    digit distribution:  {digit_dist[r]}")
        else:
            print(f"Client {r} has {total} samples with label distribution: {label_dist[r]}")
        """
        print("Top features: ", end="")
        for i in [36, 18, 24, 0, 1, 2, 34, 7]:
            print(f" {label_dist[r][i]:.4}", end="")
        print("")
        """

    print("Partition tests passed!")


def compare_tensors(t1, t2, always_print=False):
    err = float(torch.max((t1 - t2).abs()))
    same = err <= TOL
    if always_print or not same:
        print(f"Error: {err}")
    assert same
