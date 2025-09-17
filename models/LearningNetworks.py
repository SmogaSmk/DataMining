import numpy as np 
import torch
import torch.nn as nn 
from joblib import Parallel, delayed
from abc import ABC 

# Parameterized module
class LinearMultiClassification(nn.Module): 
  def __init__(self, X=None, y=None, hidden_size1 = 10, hidden_size2 = 8): 
    super(LinearMultiClassification, self).__init__()
    
    self.input_size = None 
    self.output_size = None 
    self.hidden_size1 = hidden_size1 
    self.hidden_size2 = hidden_size2
    self.network = None
    
    if X is not None and y is not None: 
      self._infer_from_data(X, y)
      self._build_network()
      
  def _build_network(self):
    if self.input_size is None or self.output_size is None:
      raise ValueError("input_size and output_size must be set before building network")

    self.network = nn.Sequential(
      nn.Linear(self.input_size, self.hidden_size1), 
      nn.ReLU(), 
      nn.Linear(self.hidden_size1, self.hidden_size2), 
      nn.ReLU(), 
      nn.Linear(self.hidden_size2, self.output_size)
    )
    
  def _infer_and_build(self, X, y=None):
    if hasattr(X, 'shape'):
      self.input_size = X.shape[1]
    elif torch.is_tensor(X):
      self.input_size = X.size(1)
    else:
      X_array = np.array(X)
      self.input_size = X_array.shape[1]
      
    if torch.is_tensor(y):
      self.output_size = len(torch.unique(y))
    elif hasattr(y, 'nunique'):
      self.output_size = y.nunique()
    else: 
      self.output_size = len(np.unique()) 
    
  def forward(self, x): 
    
    self._infer_and_build(x, y=None)
    
    return self.network(x)

  def __repr__(self): 
    if self.network is not None:
      return (f"LinearMultiClassification("
            f"input_size = {self.input_size}, "
            f"hidden_size1 = {self.hidden_size1}, "
            f"hidden_size2 = {self.hidden_size2}, "
            f"output_size = {self.output_size} )"
            )
    else: 
      return (f"LinearMultiClassification("
              f"hidden_size1={self.hidden_size1}, "
              f"hidden_size2={self.hidden_size2}, "
              f"network=NotBuilt)")
      
  