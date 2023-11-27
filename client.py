from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader



class ClientTrainer(ABC):
    def __init__(self, model, args, device):
        """
        ClientTrainer trains the model provided on the specified data partition
        Args:
            model: a NN model architecture that will be trained by the ClientTrainer.
            args: global args coming from params.py
            device: cpu or gpu (you can use cpu for this lab)

        """
        self.model = model
        self.args = args
        self.device = device 
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.client_idx = None
        self.local_training_data = None 


    def update_train_local_dataset(
        self,
        client_idx,
        local_training_data,
    ):
        """
        Set the the data partition which the ClientTrainer will use for trainer.
        Args:
            client_idx: what client-id is being simulated by ClientTrainer
            local_training_data: The data loader of the client-id's data partition
        """

        self.client_idx = client_idx
        self.local_training_data = local_training_data


    def set_model(self, model_state_dict):
        """
        Set the local model to provided params.
        Args:
          model_state_dict: Dictionary of model params that can be used to set the client's local model.
        """
        # update the weights into the model 
        self.model.load_state_dict(model_state_dict)


    def train(self, lr, local_ep, **kwargs):
        """
        Train the local model from set model on specified data partition 
        from update_train_local_dataset.
        Args:
            lr: the learning rate for intializing the optimizer for training.
            local_ep: Number of local epochs the provided model will be trained
        Returns:
            Trained model state_dict

        Please also log the clients local training as follows (an example):

        [LOCAL] Client Index = 7	Epoch: 0	Loss: 0.573473
        """

        # Move the model to the device
        self.model.to(self.device)
        self.model.train() # begin to train 

        # define the optimizaer and loss function-nn.CrossEntropyLoss() 
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        # criterion = nn.CrossEntropyLoss().to(self.device)

      # Training loop using data from update_train_local_dataset
        for epoch in range(local_ep):
            total_loss = 0.0
            for data, target in self.local_training_data:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"[LOCAL] Client Index = {self.client_idx}\tEpoch: {epoch}\tLoss: {total_loss / len(self.local_training_data)}")
            
        return self.model.state_dict()



