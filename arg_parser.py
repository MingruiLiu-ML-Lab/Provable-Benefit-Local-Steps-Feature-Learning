"""
Command-line argument parsing.
"""

import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='Distributed Training SGDClipGrad on CIFAR10 with Resnet26')
    parser.add_argument('--eta0', type=float, default=0.1,
                        help='Initial learning rate (default: 0.1).')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='Weight decay used in optimizer (default: 0).')
    parser.add_argument('--step-decay-milestones', nargs='*', default=[],
                        help='Used for step decay denoting at which rounds to decrease the step size and the clipping parameter (default: []).')
    parser.add_argument('--step-decay-factor', type=float, default=0.1,
                        help='Step size and clipping paramter shall be multiplied by this on step decay (default: 0.1).')
    parser.add_argument('--clipping-param', type=float, default=1.0,
                        help='Weight decay used in optimizer (default: 1.0).')
    parser.add_argument('--algorithm', type=str, default='local_clip',
                        choices=['local_clip', 'scaffold', 'episode', 'minibatch_clip', 'episode_mem'],
                        help='Optimization algorithm (default: local_clip).')
    parser.add_argument('--init_corrections', action="store_true", default=False,
                        help="Initialize corrections for ALL clients at the beginning of training for SCAFFOLD and EPISODE.")

    parser.add_argument('--total-clients', type=int, default=8,
                        help='Total number of clients. May be larger than world_size.')
    parser.add_argument('--participating-clients', type=int, default=8,
                        help='Number of clients in each round. Must be no larger than total_clients.')
    parser.add_argument('--client-sample-mode', type=str, default="uniform",
                        choices=["uniform", "dirichlet", "target", "periodic"],
                        help='Number of clients in each round. Must be no larger than total_clients.')
    parser.add_argument('--client-sample-param', type=str, default=None,
                        help='Parameter for client sampling, with varying purpose depending on the client sampling mode. See sample_clients() in train.py for more detail.')
    parser.add_argument('--world-size', type=int, default=8,
                        help='Number of processes in training (default: 8).')
    parser.add_argument('--rank', type=int, default=0,
                        help='Which process is this (default: 0).')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='Which GPU is used in this process (default: 0).')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of processes for data loading.')
    parser.add_argument('--init-method', type=str, default='file://',
                        help='URL specifying how to initialize the process group (default: file//).')  
    parser.add_argument('--communication-interval', type=int, default=8,
                        help='Number of local update steps (default: 8).')

    # Training
    parser.add_argument('--rounds', type=int, default=1000,
                        help='Total number of communication rounds (default: 1000).')
    parser.add_argument('--batchsize', type=int, default=32,
                        help='How many images in each train step for each GPU (default: 32).')
    parser.add_argument('--validation', action='store_true',
                        help='Do validation (True) or test (False) (default: False).')        
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Percentage of training samples used as validation (default: 0.1).')
    parser.add_argument('--num-evals', type=int, default=1,
                        help='How many times to evaluate the model during training. (default: 10).')
    parser.add_argument('--init-model', type=str, default='../logs/init_model.pth',
                        help='Path to store/load the initial weights (default: ../logs/init_model.pth).')  
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='Which dataset to run on (default: CIFAR10).')  
    parser.add_argument('--dataroot', type=str, default='../data',
                        help='Where to retrieve data (default: ../data).')
    parser.add_argument('--model', type=str, default='resnet56',
                        help='Which model to use (default: resnet56).')  
    parser.add_argument('--loss', type=str, default='svm', choices=["svm", "cross_entropy"], help="Loss function to train model.")
    parser.add_argument('--no-data-augment', default=False, action='store_true',
                        help='Turn off data augmentation.')
    parser.add_argument('--no-batch-norm', default=False, action='store_true')

    parser.add_argument("--encoder_dim", type=int, default=2048, help="encoder nhid dimension")
    parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
    parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
    parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
    parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
    parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
    parser.add_argument("--linear_fc", default=False, action='store_true', help="don't use nonlinearity in fc")
    parser.add_argument("--unidirectional", default=False, action='store_true', help="don't use bidirectional recurrent network.")
    parser.add_argument("--rnn", default=False, action='store_true', help="don't use LSTM, use vanilla RNN instead.")

    parser.add_argument('--reproducible', action='store_true',
                        help='Ensure reproducibility (default: False).')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='Random seed (default: 0).')
    parser.add_argument('--small', action='store_true',
                        help='Use mini dataset (default: False).')
    parser.add_argument('--heterogeneity', type=float, default=0.0,
                        help='Data heterogeneity level, from 0 to 1 (default: 0.0).')
    parser.add_argument('--fedprox', action='store_true',
                        help='Add FedProx regularization term. (default: False).')
    parser.add_argument('--fedprox-mu', type=float, default=0.1,
                        help='Regularization constant for FedProx (default: 0.1).')

    parser.add_argument('--log-folder', type=str, default='../logs',
                        help='Where to store results.')

    parser.add_argument('--individual-init', default=False, action="store_true")
    parser.add_argument('--mod-last-layer', default=None, choices=["random", "fixed"])
    parser.add_argument('--binarize-classes', default=False, action="store_true")
    parser.add_argument('--n_cnn_layers', type=int, default=1)
    parser.add_argument('--visualize-features', default=False, action="store_true")
    parser.add_argument('--feature-noise', type=float, default=0.0)

    return parser.parse_args()
