import params
from partition_data import load_partition_data_mnist
from models import SingleMLP
import json
from server import ServerTrainer
from client import ClientTrainer
import torch
from lr_scheduler import ConstantLrScheduler
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    args = params._parse_args()
    dataset = load_partition_data_mnist(args)

    model_args = [args.hidden_dim, args.expand_ratio, args.width_mult]
    client_model, server_model = SingleMLP(*model_args), SingleMLP(*model_args)
    client_trainer = ClientTrainer(client_model, args=args, device=device)
    lr_scheduler = ConstantLrScheduler(args.lr)

    server_trainer = ServerTrainer(
        client_trainer=client_trainer,
        global_model=server_model,
        lr_scheduler=lr_scheduler,
        dataset=dataset,
        device=device,
        args=args,
    )

    stats = dict()
    final_model = server_trainer.train(stats)
    if (args.num_rounds - 1) in stats.keys():
        print(stats[args.num_rounds - 1])
