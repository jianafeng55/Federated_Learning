import argparse
import os
import json
import torch
import random
import numpy as np
import math

parser = argparse.ArgumentParser(description="PyTorch MNIST Training")
group = parser.add_argument_group("Optimization parameters")

group.add_argument(
    "--lr", default=0.1, type=float, help="learning rate (default=0.01)"
)
group.add_argument(
    "--momentum", default=0.2, type=float, help="learning rate (default=0.9)"
)

group.add_argument(
    "--max_norm",
    default=5.0,
    type=float,
    help="Gradient Clipping for clipping gradient norms in client local training (default=1.0)",
)

group.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="batch size for single GPU (default=32)",
)
group.add_argument(
    "--optim",
    type=str,
    default="sgd",
    help="optimizer use for training: ['sgd', 'adam'] (default = 'sgd')",
)

group = parser.add_argument_group("FedAvg parameters")

group.add_argument(
    "--partition",
    type=str,
    default="homo",
    help="dataset partition: ['homo', 'hetero'] (default = 'homo')",
)

group.add_argument(
    "--local_epochs",
    type=int,
    default=2,
    help="Local epochs per client (default=1)",
)

group.add_argument(
    "--alpha",
    type=float,
    default=1.0,
    help="Drichelet Parameters for partioning the data",
)

group.add_argument(
    "--total_num_clients",
    type=int,
    default=10,
    help="Number of clients (default=10)",
)

group.add_argument(
    "--client_participation_ratio",
    type=float,
    default=0.5,
    help="Client participating in each round (default=0.5)",
)

group.add_argument(
    "--num_rounds",
    type=int,
    default=20,
    help="Total rounds for Federated Learning(default=10)",
)
group.add_argument(
    "--test_frequency",
    type=int,
    default=2,
    help="fequency to test the global model on test dataset",
)
group = parser.add_argument_group("Data parameters")
group.add_argument(
    "--data-dir",
    type=str,
    default="./data",
    help="Data directory for training MNIST dataset",
)
group.add_argument(
    "--save-dir",
    type=str,
    default="./output",
    help="Data directory for training MNIST dataset",
)

group = parser.add_argument_group("Model parameters")
group.add_argument(
    "--hidden-dim", type=int, default=50, help="hidden dimension for MLP"
)
group.add_argument(
    "--expand-ratio",
    type=float,
    default=1.6,
    help="Ratio of hidden layer input dim to output dim",
)
group.add_argument(
    "--width-mult",
    type=float,
    default=1.0,
    help="Changes the output dim by factor width-mult of every layer in the MLP",
)
parser.add_argument(
    "--seed",
    type=int,
    default=123,
    help="Seed for the run",
)


def _parse_args():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.total_clients_per_round = math.ceil(
        args.client_participation_ratio * args.total_num_clients
    )
    return args
