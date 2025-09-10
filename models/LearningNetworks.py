import numpy as np 
import torch
import torch.nn as nn 
from joblib import Parallel, delayed
from abc import ABC 

# Parameterized module
class LinearMultiClassification(nn.Module): 
  def __init__(self, input_size, output_size, hidden_size1 = 10, hidden_size2 = 8): 
    super(LinearMultiClassification, self).__init__()
    self.network = nn.Sequential(
      nn.Linear(input_size, 10), 
      nn.ReLU(), 
      nn.Linear(10, 8), 
      nn.ReLU(), 
      nn.Linear(8, output_size)
    )
    
  def forward(self, x): 
    return self.network(x)


