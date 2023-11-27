import torch.nn as nn
import torch.nn.functional as F
import math


class SingleMLP(nn.Module):
    def __init__(self, hidden_dim=50, expand_ratio=1.0, width_multiplier=1.0):
        """
        Create a 3 layer MLP model with three Fully Connected (FC) layers.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expand_ratio = expand_ratio
        self.width_multiplier = width_multiplier

        self.fc1 = nn.Linear(28*28, self.hidden_layer_input_channel)
        self.fc1_drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(self.hidden_layer_input_channel, self.hidden_layer_output_channel)
        self.fc2_drop = nn.Dropout(0.1)
        self.fc3 = nn.Linear(self.hidden_layer_output_channel, 10)
        

    @property
    def hidden_layer_input_channel(self):
        """
        Input dimension of the middle FC layer. hidden_dim, Using expand_ratio, width_multiplier
        """
        return int(self.hidden_dim * self.expand_ratio * self.width_multiplier)

    @property
    def hidden_layer_output_channel(self):
        """
        Output dimension of the middle FC layer. Using hidden_dim, width_multiplier.
        """
        return int(self.hidden_dim * self.width_multiplier)

    def forward(self, x):
        """ """
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)
