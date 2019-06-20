import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Define your neural networks in this class. 
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.

class ValueNetwork(nn.Module):
	def __init__(self):
		super(ValueNetwork, self).__init__()
		self.fc1 = nn.Linear(15, 50)
		self.fc1.weight.data.normal_(0, 0.1)   # initialization
		self.out = nn.Linear(50, 4)
		self.out.weight.data.normal_(0, 0.1)   # initialization


	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		actions_value = self.out(x)
		return actions_value

