from abc import ABC, abstractmethod
import numpy as np
import logging
import copy
import torch
import os
import torch.nn as nn

"""
federated algorithms:
M = global_model
for each round:
    ampledClients = _client_sampling(round_idx) # args.total_clients_per_round
    local_updates = []
    for client_i in sampledClients:
        Mi = train M on client_i for specified local_epochs -- (args.local_epochs)
        local_updates.append(Mi)

    Update Mg based on aggregate(local_updates) using FedAvg algorithm
    Evaluate Mg on global test dataset
"""


class ServerTrainer(ABC):
    def __init__(
        self,
        client_trainer,
        global_model,
        args,
        lr_scheduler,
        dataset,
        device="cpu",
    ) -> None:
        """
        Args:
         client_trainer: ClientTrainer class object for simulating FL training.
         global_model: This model will be trained using FedAvg algorithm.
         args: global args from params.py
         lr_scheduler: Scheduler to adjust learning rate per round.
         dataset: The partitioned dataset. 
         Tuple got from `load_partition_data_mnist` method in `partition_data.py`
         device: cpu or gpu


        """
        self.client_trainer = client_trainer
        self.global_model = global_model
        self.args = args
        self.lr_scheduler = lr_scheduler

        (self.N_traindata, self.N_testdata, self.dataloader_test, 
        self.data_local_num_dict, self.train_data_local_dict, self.class_num) = dataset


        self.device = device


    def _client_sampling(self, round_idx):
        """
        Sample k random client ids based where k=args.total_clients_per_round
        """
        np.random.seed(round_idx)
        k = max(self.args.total_clients_per_round,1) # randomly select k clients(make sure k >=1)
        rand_client = np.random.permutation(self.args.total_num_clients)[:k]
        # eg. rand_client = [4,1,0,8,2]
        return rand_client


    def _aggregate(self, w_locals, weight_coefficients):
        """
        Performs FedAvg aggregation on trained models obtained from clients.
        Args:
           w_locals: List of trained model params from sampled clients
           weight_coefficients: coefficients for weighted aggregation
        Returns:
            Final weighted aggregation.
        """

        global_state_dict  = w_locals[0]
        for key in global_state_dict:
            global_state_dict[key] = torch.div(global_state_dict[key], 1/weight_coefficients[0])
            # similar as global_state_dict[key] = w[key]* weight_coefficients[0]

        for key in global_state_dict:
            for w, weight_coefficient in zip(w_locals[1:], weight_coefficients[1:]):
                global_state_dict[key] += torch.div(w[key], 1/weight_coefficient)

        return global_state_dict


    def _get_weight_coefficient(
        self, num_local_train_sample, total_train_sample, local_clients
    ):
        """
        Get weighted coefficient for a client based on FedAvg aggregation formula.
        Args:
           num_local_train_sample: Total number of data points in entire dataset
           total_train_sample: data points in a specific clients partition
           num_local_clients: number of clients participating in a round.
        Returns:
            Return the coefficient for weighted aggregation.
        """
        # implement the paper formular: n_k/m_k instead of using the args
        sum_ = 0
        for id in local_clients:
            sum_ += self.data_local_num_dict[id]
        return  total_train_sample / sum_


    def train_one_round(self, round_num, local_ep=None, **kwargs):
        """
        Simulate on round in Federated Learning. At the end of this round, 
        the global model should be updated based on local updates from sampled clients.
        Args:
           round_num: round index or nth round that's being performed.
           local_ep: local epochs that each clients need to perform.
        """
        sampled_clients = self._client_sampling(round_num) #round_num = round_idx
        #Mg <- global_model
        self.client_trainer.set_model(self.global_model.state_dict())

        local_updates = []
        for client_id in sampled_clients:
            self.client_trainer.update_train_local_dataset(client_id, self.train_data_local_dict[client_id])
            model_state_dict = self.client_trainer.train(lr=self.args.lr, local_ep=local_ep, **kwargs)

            local_updates.append(copy.deepcopy(model_state_dict))

        weight_coefficients = [
            self._get_weight_coefficient(
                len(self.train_data_local_dict[client_id]),
                self.data_local_num_dict[client_id],
                sampled_clients, 
            )
            for client_id in sampled_clients
        ]
        #aggregate the weights 
        global_state_dict = self._aggregate(local_updates, weight_coefficients)
        #update the global model
        self.global_model.load_state_dict(global_state_dict)


    def train(self, per_round_stats):
        """
        Perform entire FL simulation for specified number of rounds. 
        Perform Eval of global model after every mod of args.test_frequency rounds. In the logs, report the test accuracy as follows as an example:
        [GLOBAL] Round: 18 Acc: 0.4825

        Args:
            per_round_stats: Append the per round test accuracy on global test dataset achieved by the global model.

        """
        for round_num in range(1, self.args.num_rounds + 1):
            #begin to train 
            self.train_one_round(round_num, local_ep=self.args.local_epochs)

            #do the inferrence 
            if round_num % self.args.test_frequency == 0:
                self.global_model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in self.dataloader_test:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.global_model(data)
                        _, predicted = torch.max(output.data, 1)
                        pred_labels = predicted.view(-1)
                        correct += torch.sum(torch.eq(pred_labels, target)).item()
                        total += target.size(0)
                    test_acc = correct / total

                print(f"[GLOBAL] Round: {round_num} Acc: {test_acc}")


