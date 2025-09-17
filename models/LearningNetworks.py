import numpy as np 
import torch
import torch.nn as nn 
from joblib import Parallel, delayed
from abc import ABC 

# Parameterized module
class LinearMultiClassificationH(nn.Module): 
  def __init__(self, X=None, y=None, hidden_size1 = 10, hidden_size2 = 8): 
    super(LinearMultiClassificationH, self).__init__()
    
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
    
  def _infer_from_data(self, X, y=None):
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
      self.output_size = len(np.unique(y)) 
    
  def forward(self, x): 
    
    self._infer_from_data(x, y=None)
    
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


class LinearMultiClassificationS(nn.Module): 
  def __init__(self, X=None, y=None): 
    super(LinearMultiClassificationS, self).__init__()
    
    self.input_size = None 
    self.output_size = None 
    self.network = None
    
    if X is not None and y is not None: 
      self._infer_from_data(X, y)
      self._build_network()
      
  def _build_network(self):
    if self.input_size is None or self.output_size is None:
      raise ValueError("input_size and output_size must be set before building network")

    self.network = nn.Sequential(
      nn.Linear(self.input_size, self.output_size), 
    )
    
  def _infer_from_data(self, X, y=None):
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
      self.output_size = len(np.unique(y)) 
    
  def forward(self, x): 
    
    self._infer_from_data(x, y=None)
    
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

class RidgeRegression(nn.Module):
  def __init__(self, X=None, y=None, alpha=1.0):
    super(RidgeRegression, self).__init__()
        
    self.input_size = None
    self.output_size = 1 
    self.alpha = alpha
    self.network = None
        
    if X is not None:
      self._infer_from_data(X, y)
      self._build_network()
    
  def _build_network(self):
    if self.input_size is None:
      raise ValueError("input_size must be set before building network")
        
    self.network = nn.Linear(self.input_size, self.output_size)
    
  def _infer_from_data(self, X, y=None):
    """从数据推断输入和输出维度"""
    if hasattr(X, 'shape'):
      self.input_size = X.shape[1]
    elif torch.is_tensor(X):
      self.input_size = X.size(1)
    else:
      X_array = np.array(X)
      self.input_size = X_array.shape[1]
        
    # 智能推断输出维度：分类任务 vs 回归任务
    if y is not None:
      if torch.is_tensor(y):
        unique_vals = torch.unique(y)
        # 如果y是整数且值的数量较少，认为是分类任务
        if y.dtype in [torch.long, torch.int32, torch.int64] or len(unique_vals) <= 20:
          self.output_size = len(unique_vals)
        else:
          self.output_size = 1  # 回归任务
      elif hasattr(y, 'nunique'):
        if y.nunique() <= 20:  # 分类任务
          self.output_size = y.nunique()
        else:
          self.output_size = 1  # 回归任务
      else:
        unique_vals = np.unique(y)
        if len(unique_vals) <= 20:  # 分类任务
          self.output_size = len(unique_vals)
        else:
          self.output_size = 1  # 回归任务
    
  def forward(self, x):
    if self.network is None:
      self._infer_from_data(x)
      self._build_network()
        
    return self.network(x)
    
  def get_l2_penalty(self):
    l2_reg = 0.0
    for param in self.parameters():
      l2_reg += torch.sum(param ** 2)
    return self.alpha * l2_reg
    
  def __repr__(self):
    if self.network is not None:
      task_type = "Classification" if self.output_size > 1 else "Regression"
      return (f"Ridge{task_type}("
        f"input_size={self.input_size}, "
        f"output_size={self.output_size}, "
        f"alpha={self.alpha})")
    else:
      return (f"RidgeRegression("
        f"alpha={self.alpha}, "
        f"network=NotBuilt)")
      
  