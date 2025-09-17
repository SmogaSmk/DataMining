import numpy as np 
import torch
import torch.nn as nn 
from joblib import Parallel, delayed
from abc import ABC
from sklearn.utils.class_weight import compute_class_weight

def _auto_compute_class_weights(y, method='balanced'):
  """
  自动计算类别权重以处理不平衡数据
  内部函数，用户无需调用
  """
  if y is None:
    return None
    
  if torch.is_tensor(y):
    y_np = y.cpu().numpy()
  elif hasattr(y, 'values'):  # pandas Series
    y_np = y.values
  else:
    y_np = np.array(y)
    
  classes = np.unique(y_np)
    
  if len(classes) <= 1:
    return None
    
  class_counts = np.bincount(y_np)
  max_count = np.max(class_counts[class_counts > 0])
  min_count = np.min(class_counts[class_counts > 0])
    
  if max_count / min_count <= 2.0:
    return None
  
  if method == 'sqrt': 
    weights = np.sqrt(len(y_np)/ (len(classes) * class_counts[classes]))
  elif method == 'cbrt': 
    weights = np.cbrt(len(y_np)/ (len(classes) * class_counts[classes]))
  elif method == 'log': 
    weights = np.log(len(y_np)/ (len(classes) * class_counts[classes]) + 1) + 1
  elif method == 'exp':
    sigma = 0.9
    weights = np.power(len(y_np)/(len(classes) * class_counts[classes]), sigma)
  else: 
    weights = compute_class_weight('balanced', classes=classes, y=y_np)
  return torch.FloatTensor(weights) 

# Parameterized module
class LinearMultiClassificationH(nn.Module): 
  def __init__(self, X=None, y=None, hidden_size1 = 10, hidden_size2 = 8, class_weight_method='auto'): 
    super(LinearMultiClassificationH, self).__init__()
    
    self.input_size = None 
    self.output_size = None 
    self.hidden_size1 = hidden_size1 
    self.hidden_size2 = hidden_size2
    self.class_weights_tensor = None
    self.class_weight_method = class_weight_method  # 新增参数
    self.network = None
    
    if X is not None and y is not None: 
      self._infer_from_data(X, y)
      self._build_network()
      self._compute_class_weights(y)
      
  def _compute_class_weights(self, y):
    if self.class_weight_method is None: 
      self.class_weights_tensor = None 
    elif isinstance(self.class_weight_method, dict): 
      weights = [self.class_weight_method.get(i, 1.0) for i in range(self.output_size)]
      self.class_weights_tensor = torch.FloatTensor(weights)
    else: 
      self.class_weights_tensor = _auto_compute_class_weights(y, method=self.class_weight_method)
      
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

  def get_class_weights(self, device=None):
    """
    获取类别权重张量，用于训练时的损失函数
    
    Args:
        device: 目标设备 (cuda/cpu)
    
    Returns:
        torch.Tensor or None: 类别权重张量
    """
    if self.class_weights_tensor is None:
      return None
    
    weights = self.class_weights_tensor.clone()
    if device is not None:
      weights = weights.to(device)
    return weights

  def __repr__(self): 
    if self.network is not None:
      auto_balanced = " (auto-balanced)" if self.class_weights_tensor is not None else ""
      return (f"LinearMultiClassificationH("
            f"input_size = {self.input_size}, "
            f"hidden_size1 = {self.hidden_size1}, "
            f"hidden_size2 = {self.hidden_size2}, "
            f"output_size = {self.output_size}{auto_balanced} )"
            )
    else: 
      return (f"LinearMultiClassificationH("
              f"hidden_size1={self.hidden_size1}, "
              f"hidden_size2={self.hidden_size2}, "
              f"network=NotBuilt)")


class LinearMultiClassificationS(nn.Module): 
  def __init__(self, X=None, y=None, class_weight_method='auto'): 
    super(LinearMultiClassificationS, self).__init__()
    
    self.input_size = None 
    self.output_size = None 
    self.class_weights_tensor = None
    self.class_weight_method = class_weight_method  # 新增参数
    self.network = None
    
    if X is not None and y is not None: 
      self._infer_from_data(X, y)
      self._build_network()
      self._compute_class_weights(y)
      
  def _compute_class_weights(self, y):
    if self.class_weight_method is None: 
      self.class_weights_tensor = None 
    elif isinstance(self.class_weight_method, dict): 
      weights = [self.class_weight_method.get(i, 1.0) for i in range(self.output_size)]
      self.class_weights_tensor = torch.FloatTensor(weights)
    else: 
      self.class_weights_tensor = _auto_compute_class_weights(y, method=self.class_weight_method)
      
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

  def get_class_weights(self, device=None):
    """
    获取类别权重张量，用于训练时的损失函数
    
    Args:
        device: 目标设备 (cuda/cpu)
    
    Returns:
        torch.Tensor or None: 类别权重张量
    """
    if self.class_weights_tensor is None:
      return None
    
    weights = self.class_weights_tensor.clone()
    if device is not None:
      weights = weights.to(device)
    return weights

  def __repr__(self): 
    if self.network is not None:
      auto_balanced = " (auto-balanced)" if self.class_weights_tensor is not None else ""
      return (f"LinearMultiClassificationS("
            f"input_size = {self.input_size}, "
            f"output_size = {self.output_size}{auto_balanced} )"
            )
    else: 
      return (f"LinearMultiClassificationS("
              f"network=NotBuilt)")

class RidgeRegression(nn.Module):
  def __init__(self, X=None, y=None, alpha=1.0, class_weight_method='auto'):
    super(RidgeRegression, self).__init__()
        
    self.input_size = None
    self.output_size = 1 
    self.alpha = alpha
    self.class_weights_tensor = None
    self.class_weight_method = class_weight_method  # 新增参数
    self.is_classification = False
    self.network = None
        
    if X is not None:
      self._infer_from_data(X, y)
      self._build_network()
      if y is not None and self.is_classification:
        self._compute_class_weights(y)
    
  def _compute_class_weights(self, y):
    if self.class_weight_method is None: 
      self.class_weights_tensor = None 
    elif isinstance(self.class_weight_method, dict): 
      weights = [self.class_weight_method.get(i, 1.0) for i in range(self.output_size)]
      self.class_weights_tensor = torch.FloatTensor(weights)
    else: 
      self.class_weights_tensor = _auto_compute_class_weights(y, method=self.class_weight_method)
    
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
          self.is_classification = True
        else:
          self.output_size = 1  # 回归任务
          self.is_classification = False
      elif hasattr(y, 'nunique'):
        if y.nunique() <= 20:  # 分类任务
          self.output_size = y.nunique()
          self.is_classification = True
        else:
          self.output_size = 1  # 回归任务
          self.is_classification = False
      else:
        unique_vals = np.unique(y)
        if len(unique_vals) <= 20:  # 分类任务
          self.output_size = len(unique_vals)
          self.is_classification = True
        else:
          self.output_size = 1  # 回归任务
          self.is_classification = False
    
  def forward(self, x):
    if self.network is None:
      self._infer_from_data(x)
      self._build_network()
        
    return self.network(x)
    
  def get_class_weights(self, device=None):
    """
    获取类别权重张量，用于训练时的损失函数（仅用于分类任务）
    
    Args:
        device: 目标设备 (cuda/cpu)
    
    Returns:
        torch.Tensor or None: 类别权重张量
    """
    if not self.is_classification or self.class_weights_tensor is None:
      return None
    
    weights = self.class_weights_tensor.clone()
    if device is not None:
      weights = weights.to(device)
    return weights
    
  def get_l2_penalty(self):
    l2_reg = 0.0
    for param in self.parameters():
      l2_reg += torch.sum(param ** 2)
    return self.alpha * l2_reg
    
  def __repr__(self):
    if self.network is not None:
      task_type = "Classification" if self.output_size > 1 else "Regression"
      auto_balanced = " (auto-balanced)" if self.is_classification and self.class_weights_tensor is not None else ""
      return (f"Ridge{task_type}("
        f"input_size={self.input_size}, "
        f"output_size={self.output_size}, "
        f"alpha={self.alpha}{auto_balanced})")
    else:
      return f"RidgeRegression(alpha={self.alpha}, network=NotBuilt)"
      
  