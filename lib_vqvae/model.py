import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    """
    Inputs:
    - input_dim : the input dimension
    - n : the number of codebooks an embedding is split into
    - e_dim : the dim of each codebook
    - hidden_dim : the structure of hidden layers
    
    input_data:
    - (batch_size, input_dim)
    output_data:
    - (batch_size, n*e_dim)
    """
    
    def __init__(self, input_dim, n, e_dim, hidden_dim, dropout_rate):
        super(Encoder, self).__init__()    
        self.input_dim = input_dim
        self.n = n
        self.e_dim = e_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        layers = []
        
        # Create hidden layers
        indim = self.input_dim
        for h_dim in self.hidden_dim:
            layers.append(nn.Linear(indim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            indim = h_dim
        # Create output layer
        layers.append(nn.Linear(indim, n*e_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x
    

class Decoder(nn.Module):
    """
    Inputs:
    - output_dim : the output dimension
    - n : the number of codebooks an embedding is split into
    - e_dim : the dim of each codebook
    - hidden_dim : the structure of hidden layers
    
    input_data:
    - (batch_size, n*e_dim)
    output_data:
    - (batch_size, output_dim)
    """
    
    def __init__(self, output_dim, n, e_dim, hidden_dim, dropout_rate):
        super(Decoder, self).__init__()    
        self.output_dim = output_dim
        self.n = n
        self.e_dim = e_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        layers = []
        
        # Create hidden layers
        indim = self.n*self.e_dim
        for h_dim in self.hidden_dim:
            layers.append(nn.Linear(indim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            indim = h_dim
        # Create output layer
        layers.append(nn.Linear(indim, self.output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x