"""
Train a model on the training set.
"""
import copy
import time
from math import floor, ceil

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from evaluate import evaluate, predict
from sgd_clip import SGDClipGrad
from data_loader import get_num_classes
from utils import kl_div, get_client_label_dist, trainable_parameters
from test import test_corrections, test_minibatch, test_initial_corrections


DEBUG = False


# Model Averaging
def average_model(world_size, model, local_models, group):

    # Average local models for clients simulated by this process.
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            p.data = torch.mean(
                torch.stack(
                    [local_models[c][i] for c in range(len(local_models))],
                ),
                dim=0
            )

    # Average models across processes.
    for param in model.parameters():
        dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM, group=group)
        param.data /= world_size
        dist.broadcast(param.data, src=0, group=group)

def average_list(world_size, l, group, sum=False):
    for param in l:
        dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM, group=group)
        if not sum:
            param /= world_size
        dist.broadcast(param.data, src=0, group=group)

def comp_grad_l2_norm(model) -> float:
    grad_l2_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_l2_norm_sq += torch.sum(param.grad.data * param.grad.data)
    grad_l2_norm = torch.sqrt(grad_l2_norm_sq).item()
    return grad_l2_norm


def train(args, train_loader, test_loader, extra_loader, net, criterion, group):
    """
    Args:
        args: parsed command line arguments.
        train_loader: an iterator over the training set.
        test_loader: an iterator over the test set.
        net: the neural network model employed.
        criterion: the loss function.

    Outputs:
        All training losses, training accuracies, test losses, and test
        accuracies on each evaluation during training.
    """

    # Initialize optimizer.
    optimizer = SGDClipGrad(
        params=trainable_parameters(net),
        lr=args.eta0,
        weight_decay=args.weight_decay,
        clipping_param=args.clipping_param,
        algorithm=args.algorithm
    )

    # Initialize settings for distributed training.
    world_size = args.world_size
    if args.participating_clients % args.world_size != 0:
        raise ValueError("participating_clients must be a multiple of world_size.")
    if args.participating_clients > args.total_clients:
        raise ValueError("participating_clients cannot be greater than total_clients.")
    if args.world_size > args.total_clients:
        raise ValueError("world_size cannot be greater than total_clients.")
    round_clients_per_worker = args.participating_clients // args.world_size
    start = floor(args.rank * args.total_clients / args.world_size)
    end = floor((args.rank + 1) * args.total_clients / args.world_size)
    local_eval_clients = list(range(start, end))

    # Determine what rounds during which to run evaluation.
    eval_rounds = [
        round((i+1) * (args.rounds-1) / args.num_evals) for i in range(args.num_evals)
    ]

    # Setup for various client sampling modes.
    client_dist = None
    sample_param = None
    if args.client_sample_mode == "dirichlet":

        # Determine a distribution from which to sample clients for participation.
        assert args.client_sample_param is not None
        alpha = float(args.client_sample_param)
        client_dist = np.random.dirichlet(alpha=[alpha] * args.total_clients)
        sample_param = alpha

    elif args.client_sample_mode == "target":

        TARGET_TEMPERATURE = 5.0

        # Determine a distribution from which to sample clients for participation. The
        # sampling probability for each client is inversely proportional to the
        # KL-divergence between that client's label distribution and some target
        # distribution over labels. The target distribution is a convex combination of
        # the client label distributions, where the weights of the combination are drawn
        # from a Dirichlet distribution.
        assert args.client_sample_param is not None
        num_classes = get_num_classes(args.dataset)
        alpha = float(args.client_sample_param)
        client_mix = np.random.dirichlet(alpha=[alpha] * args.total_clients)
        client_mix = torch.tensor(client_mix).unsqueeze(-1)
        client_label_dist = get_client_label_dist(train_loader, num_classes)
        target_dist = torch.sum(client_mix * client_label_dist, dim=0)

        dist_diffs = torch.tensor([kl_div(l, target_dist) for l in client_label_dist])
        #client_dist = 1.0 / (dist_diffs + 1e-9)
        client_dist = -dist_diffs
        #client_dist = client_dist / torch.sum(client_dist)
        client_dist = F.softmax(TARGET_TEMPERATURE * client_dist, dim=0)
        client_dist = client_dist.numpy()
        sample_param = alpha

    elif args.client_sample_mode == "periodic":

        # Parse client_sample_param.
        assert args.client_sample_param is not None
        assert "," in args.client_sample_param
        comma_pos = args.client_sample_param.index(",")
        avail_clients = int(args.client_sample_param[:comma_pos])
        switch_rounds = int(args.client_sample_param[comma_pos+1:])
        if args.total_clients % avail_clients != 0:
            raise ValueError(
                "Number of simultaneously available clients in periodic sampling must be a divisor of the total number of clients."
            )
        sample_param = [avail_clients, switch_rounds]

    if client_dist is not None:
        client_dist = torch.tensor(client_dist).cuda()
        dist.broadcast(client_dist, src=0, group=group)
        uniform_dist = torch.ones_like(client_dist) / torch.numel(client_dist)

        div_from_uniform = float(kl_div(client_dist, uniform_dist))
        client_dist = client_dist.cpu().numpy()

        f_client_dist = "["
        for c in range(args.total_clients):
            f_client_dist += f"{client_dist[c]:.4f}"
            if c < args.total_clients - 1:
                f_client_dist += ", "
        f_client_dist += "]"
        print(f"Client dist: {f_client_dist}")
        print(f"Max, min client p: {float(np.max(client_dist)):.5f}, {float(np.min(client_dist)):.5f}")
        print(f"KL divergence from uniform: {div_from_uniform}")

    prev_params = None
    local_corrections = [None] * args.total_clients
    global_correction = None
    round_avg_grad = []

    # If necessary, initialize corrections for ALL clients.
    if args.init_corrections:

        if args.algorithm not in ["scaffold", "episode_mem"]:
            raise ValueError(
                "Argument --init_corrections only valid when algorithm='scaffold' or"
                "algorithm='episode_mem'."
            )

        # Compute local correction for local clients.
        n_params = None
        for client in local_eval_clients:
            train_loader.set_clients([client])
            for i in range(args.communication_interval):

                # Sample batch and compute gradient for local correction.
                net.zero_grad()
                data = train_loader.next()
                inputs, labels = data
                labels = labels.cuda()
                outputs = predict(net, inputs, args.dataset)
                loss = criterion(outputs, labels)
                loss.backward()
                current_grad = [
                    p.grad.clone().detach() for p in trainable_parameters(net) if p.grad is not None
                ]
                if n_params is None:
                    n_params = len(current_grad)
                    global_correction = [0] * n_params
                if local_corrections[client] is None:
                    local_corrections[client] = [0] * n_params

                # Add current gradient to running average.
                for j in range(n_params):
                    local_corrections[client][j] += current_grad[j]
                    global_correction[j] += current_grad[j]

            for j in range(n_params):
                local_corrections[client][j] /= args.communication_interval

        for j in range(len(global_correction)):
            global_correction[j] /= args.communication_interval

        # Share local corrections across workers.
        ex_c = local_eval_clients[0]
        for client in range(args.total_clients):
            src = ceil((client + 1) / (args.total_clients / args.world_size) - 1)
            if local_corrections[client] is None:
                local_corrections[client] = [
                    torch.zeros_like(local_corrections[ex_c][j]) for j in range(n_params)
                ]
            for j in range(n_params):
                dist.broadcast(local_corrections[client][j], src=src, group=group)

        # Average local corrections to get global correction.
        average_list(world_size, global_correction, group, sum=True)
        for j in range(len(global_correction)):
            global_correction[j] /= args.total_clients

        if DEBUG:
            test_initial_corrections(local_corrections, global_correction, args.rank, group)

    eval_local_train_losses = []
    eval_local_train_accuracies = []
    local_train_losses = []
    local_train_accuracies = []
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    eval_clip_ops = []
    clip_ops = []
    eval_elasped_times = []
    eval_start = time.time()
    for r in range(args.rounds):
        net.train()

        # Store network from beginning of round.
        prev_params = [
            p.clone().detach() for p in trainable_parameters(net)
        ]

        # Sample clients for this round.
        round_clients = sample_clients(
            total_clients=args.total_clients,
            participating_clients=args.participating_clients,
            sample_mode=args.client_sample_mode,
            sample_param=sample_param,
            r=r,
            group=group,
            client_dist=client_dist,
        )

        # Update corrections, if necessary.
        if args.algorithm == "episode":
            local_corrections, global_correction = episode_correction(
                net,
                args.dataset,
                round_clients,
                args.total_clients,
                args.rank,
                extra_loader,
                criterion,
                world_size,
                group
            )

        # Train a local model for each simulated client.
        updated_client_params = []
        round_avg_grad = []
        for i in range(round_clients_per_worker):

            client = round_clients[args.rank * round_clients_per_worker + i]
            net, local_losses, local_accs, local_clip_ops, local_avg_grad = train_local_model(
                net,
                train_loader,
                optimizer,
                criterion,
                client,
                local_corrections[client],
                global_correction,
                prev_params,
                args,
            )
            eval_local_train_losses += local_losses
            eval_local_train_accuracies += local_accs
            eval_clip_ops += local_clip_ops

            # Store updated client parameters.
            updated_client_params.append(
                [p.clone().detach().data for p in net.parameters()]
            )

            # Store average of gradient over current round, and reset parameters of
            # `net` to previous global model.
            with torch.no_grad():
                if args.algorithm in ["scaffold", "episode_mem"]:
                    round_avg_grad.append([])
                    for g in local_avg_grad:
                        round_avg_grad[i].append(g.clone())
                elif args.algorithm == "minibatch_clip":
                    if len(round_avg_grad) == 0:
                        round_avg_grad = [0] * len(local_avg_grad)
                    for j, g in enumerate(local_avg_grad):
                        round_avg_grad[j] += g.clone() / round_clients_per_worker
                for (p, prev_p) in zip(trainable_parameters(net), prev_params):
                    p.data = prev_p.data

        # Perform one minibatch update if necessary, otherwise average local models.
        with torch.no_grad():
            if args.algorithm == "minibatch_clip":
                clip = minibatch_update(optimizer, round_avg_grad, world_size, group)
                eval_clip_ops.append(clip)
                if DEBUG:
                    test_minibatch(net, args.rank, group)
            else:
                average_model(world_size, net, updated_client_params, group)

        # Update corrections, if necessary.
        if args.algorithm in ["scaffold", "episode_mem"]:
            local_corrections, global_correction = scaffold_correction(
                local_corrections,
                global_correction,
                round_avg_grad,
                round_clients,
                args.total_clients,
                args.rank,
                world_size,
                group
            )

        # Evaluate the model on training and validation dataset.
        if r in eval_rounds:
            elapsed_time = time.time() - eval_start
            eval_elasped_times.append(elapsed_time)

            local_train_loss = np.mean(eval_local_train_losses)
            local_train_accuracy = np.mean(eval_local_train_accuracies)
            local_train_losses.append(local_train_loss)
            local_train_accuracies.append(local_train_accuracy)
            eval_local_train_losses = []
            eval_local_train_accuracies = []

            train_loader.set_clients(local_eval_clients)
            train_loss, train_accuracy = evaluate(train_loader, net, criterion, args.dataset)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            test_loss, test_accuracy = evaluate(test_loader, net, criterion, args.dataset)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            clip_ops.append(np.mean(eval_clip_ops))
            eval_clip_ops = []

            total_time = time.time() - eval_start

            print(f'Rank {args.rank} '
                  f'| GPU {args.gpu_id} '
                  f'| Round {r+1}/{args.rounds} '
                  f'| train time {elapsed_time:.2f} s '
                  f'| total time {total_time:.2f} s '
                  f'| local train loss {local_train_loss:.4f} '
                  f'| local train acc {local_train_accuracy:.4f} '
                  f'| train loss {train_loss:.4f} '
                  f'| train acc {train_accuracy:.4f} '
                  f'| test loss {test_loss:.4f} '
                  f'| test acc {test_accuracy:.4f}')

            eval_start = time.time()

        if str(r+1) in args.step_decay_milestones:
            print(f'Decay step size and clip param at Round {r+1}.')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.step_decay_factor
                if 'clipping_param' in param_group:
                    param_group['clipping_param'] *= args.step_decay_factor

    train_results= {
        'local_train_losses': local_train_losses,
        'local_train_accuracies': local_train_accuracies,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'eval_elasped_times': eval_elasped_times,
        'clip_ops': clip_ops
    }
    return train_results, net


def train_local_model(
    net,
    train_loader,
    optimizer,
    criterion,
    current_client,
    local_correction,
    global_correction,
    prev_params,
    args,
):
    """ Train a single client model for one communication round. """

    # Set client of local worker.
    train_loader.set_clients([current_client])

    round_avg_grad = None
    round_losses = []
    round_accuracies = []
    round_clip_ops = []
    for _ in range(args.communication_interval):

        # Compute loss.
        optimizer.zero_grad()
        data = train_loader.next()
        inputs, labels = data
        labels = labels.cuda()
        outputs = predict(net, inputs, args.dataset)
        loss = criterion(outputs, labels)

        # Compute regularization term for FedProx.
        if args.fedprox:
            params = list(trainable_parameters(net))
            assert len(params) == len(prev_params)
            for p, gp in zip(params, prev_params):
                loss += args.fedprox_mu * torch.sum((p - gp) ** 2) / 2.0

        loss.backward()

        # Update average gradient of current round.
        if args.algorithm in ["scaffold", "episode_mem", "minibatch_clip"]:
            round_avg_grad = update_avg_grad(
                round_avg_grad, net, args.communication_interval
            )

        # Take update step. For minibatch_clip, parameters aren't updated until the end
        # of the round.
        if args.algorithm != "minibatch_clip":
            _, clip_operation = optimizer.step(
                local_correction=local_correction,
                global_correction=global_correction,
            )
            round_clip_ops.append(clip_operation)

        # Track training metrics.
        _, predicted = torch.max(outputs, 1)
        accuracy = (1.0 * (predicted == labels)).mean().item()
        round_losses.append(loss.item())
        round_accuracies.append(accuracy)

    return net, round_losses, round_accuracies, round_clip_ops, round_avg_grad


def minibatch_update(optimizer, round_avg_grad, world_size, group):
    """ Single update step for minibatch clipping. """

    # Compute average gradient across all participating clients.
    average_list(world_size, round_avg_grad, group)

    # Compute norm of update.
    updates = []
    local_update_l2_norm_sq = 0.0
    i = 0
    for group in optimizer.param_groups:
        weight_decay = group['weight_decay']

        for p in group['params']:
            if p.grad is None:
                continue

            d_p = round_avg_grad[i].clone().detach()
            if weight_decay != 0:
                d_p = d_p.add(p, alpha=weight_decay)
            updates.append(d_p.clone().detach())
            local_update_l2_norm_sq += torch.sum(d_p.data * d_p.data)
            i += 1
    local_update_l2_norm = torch.sqrt(local_update_l2_norm_sq).item()

    # Apply update to each parameter.
    clip = local_update_l2_norm > group['clipping_param'] / group['lr']
    lr = group['lr']
    if clip:
        lr = group['clipping_param'] / (1e-10 + local_update_l2_norm)
    i = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            p.add_(updates[i], alpha=-lr)
            i += 1

    return clip


def scaffold_correction(
    local_corrections,
    global_correction,
    round_avg_grad,
    round_clients,
    total_clients,
    rank,
    world_size,
    group
):
    """ Compute local and global corrections for SCAFFOLD. """

    round_clients_per_worker = len(round_avg_grad)
    n_params = len(round_avg_grad[0])

    # Handle uninitialized corrections.
    for i in range(len(local_corrections)):
        if local_corrections[i] is None:
            local_corrections[i] = [
                torch.zeros_like(round_avg_grad[0][j]) for j in range(n_params)
            ]
    if global_correction is None:
        global_correction = [
            torch.zeros_like(round_avg_grad[0][j]) for j in range(n_params)
        ]

    # Store original local corrections for testing.
    if DEBUG:
        prev_local_corrections = [[c.clone() for c in lc] for lc in local_corrections]

    # Compute delta of local corrections for sampled clients.
    delta_local_corrections = [[] for _ in range(n_params)]
    with torch.no_grad():
        for i in range(round_clients_per_worker):
            client = round_clients[rank * round_clients_per_worker + i]
            for j, g in enumerate(round_avg_grad[i]):
                delta_local_corrections[j].append(g - local_corrections[client][j])
    for j in range(n_params):
        delta_local_corrections[j] = torch.stack(delta_local_corrections[j])

    # Communicate and apply change in local corrections for sampled clients.
    all_deltas = [
        [torch.zeros_like(delta_local_corrections[j]) for _ in range(world_size)]
        for j in range(n_params)
    ]
    with torch.no_grad():
        for j in range(n_params):

            # Gather delta of local corrections.
            dist.all_gather(all_deltas[j], delta_local_corrections[j], group=group)

            # Apply delta to local corrections and global correction.
            for r in range(world_size):
                for i in range(round_clients_per_worker):
                    k = r * round_clients_per_worker + i
                    client = round_clients[k]
                    local_corrections[client][j] += all_deltas[j][r][i]
                    global_correction[j] += all_deltas[j][r][i] / total_clients

    # Run tests for local and global corrections.
    if DEBUG:
        with torch.no_grad():
            test_corrections(
                local_corrections,
                global_correction,
                prev_local_corrections,
                round_avg_grad,
                round_clients,
                rank,
                group,
            )

    return local_corrections, global_correction


def update_avg_grad(round_avg_grad, net, communication_interval):
    """ Update average gradient of current round. """

    current_grad = [p.grad for p in trainable_parameters(net) if p.grad is not None]
    with torch.no_grad():
        if round_avg_grad is None:
            round_avg_grad = [0] * len(current_grad)
        assert len(current_grad) == len(round_avg_grad)
        for i in range(len(current_grad)):
            round_avg_grad[i] += current_grad[i] / communication_interval

    return round_avg_grad


def episode_correction(
    net, dataset_name, round_clients, total_clients, rank, extra_loader, criterion, world_size, group
):
    """ Compute local and global corrections for EPISODE. """

    round_clients_per_worker = len(round_clients) // world_size

    # Compute local correction for each client.
    local_corrections = [None] * total_clients
    global_correction = None
    for i in range(round_clients_per_worker):

        client = round_clients[rank * round_clients_per_worker + i]
        extra_loader.set_clients([client])

        # Sample batch and compute gradient for local correction.
        net.zero_grad()
        data = extra_loader.next()
        inputs, labels = data
        labels = labels.cuda()
        outputs = predict(net, inputs, dataset_name)
        loss = criterion(outputs, labels)
        loss.backward()
        local_corrections[client] = [
            p.grad.clone().detach() for p in trainable_parameters(net) if p.grad is not None
        ]
        if global_correction is None:
            global_correction = [0] * len(local_corrections[client])
        for j, g in enumerate(local_corrections[client]):
            global_correction[j] += g.clone() / round_clients_per_worker

    # Average local corrections to get global correction.
    average_list(world_size, global_correction, group)

    return local_corrections, global_correction


def sample_clients(
    total_clients: int,
    participating_clients: int,
    sample_mode: str,
    sample_param,
    r: int,
    group,
    client_dist: np.array = None,
) -> torch.Tensor:
    """
    Sample clients to participate in this round and broadcast to all workers.

    Sample modes:
        uniform: Sample clients uniformly.
        dirichlet: Sample clients from a distribution sampled from a Dirichlet
            distribution with parameter alpha=sample_param.
        target: Sample clients from a distribution constructed so that a client's
            probablility of being sampled is inversely proportional to the KL divegence
            between that client's label distribution and same target distribution of
            labels.
        periodic: Client availability is periodic. Here
            sample_param=[avail_clients,switch_rounds], where in each section
            `avail_clients` are available, availability switches after `switch_rounds`
            rounds.
    """

    # Sample clients based on sampling mode.
    if sample_mode == "uniform":
        round_clients = np.random.choice(
            np.arange(total_clients),
            size=participating_clients,
            replace=False,
        )
    elif sample_mode in ["dirichlet", "target"]:
        assert client_dist is not None
        round_clients = np.random.choice(
            np.arange(total_clients),
            size=participating_clients,
            replace=False,
            p=client_dist,
        )
    elif sample_mode == "periodic":
        avail_clients, switch_rounds = sample_param
        switch_period = total_clients // avail_clients
        switch_idx = (r // switch_rounds) % switch_period
        start_client = switch_idx * avail_clients
        end_client = (switch_idx + 1) * avail_clients
        round_clients = np.random.choice(
            np.arange(start=start_client, stop=end_client),
            size=participating_clients,
            replace=False
        )
    else:
        raise NotImplementedError

    # Sort the indices of the sampled clients so that the same clients stay on the same
    # worker in the case of full participation.
    round_clients = np.sort(round_clients)

    # Broadcast sampled indices to all workers.
    round_clients = torch.tensor(round_clients).cuda()
    dist.broadcast(round_clients, src=0, group=group)
    round_clients = [int(c) for c in round_clients]

    return round_clients
