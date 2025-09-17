import numpy as np 
import torch 
import pandas as pd

class TensorConverter: 
  '''
  Tensor Converter 
  Dealing with tensor tasks transform
  '''
  @ staticmethod
  def to_numpy(data): 
    '''Transform other data (especially Pytorch tensor) to Numpy array'''
    if torch.is_tensor(data): 
      return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray): 
      return data 
    else: 
      return np.array(data)
    
  @ staticmethod 
  def to_tensor(data, device = 'cuda'): 
    '''Transform python origin, numpy, pandas or other data structure to tensor'''
    if torch.is_tensor(data): 
      return data.to(device)
    else: 
      # 确保数据类型为 float32
      np_data = np.array(data, dtype=np.float32)
      return torch.from_numpy(np_data).to(device)
    
  @ staticmethod 
  def ensure_2d(data): 
    '''Make sure the data's dimension(especially 1 dim data) is 2'''
    if isinstance(data, np.ndarray):
      return data.reshape(1, -1) if data.ndim == 1 else data
    elif torch.is_tensor(data): 
      return data.view(1, -1) if data.ndim == 1 else data
    else: 
      arr = np.array(data)
      return arr.reshape(1, -1) if arr.ndim == 1 else arr


class TensorDataProcessor:
    """扩展的张量数据处理器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.converter = TensorConverter()
    
    def prepare_features(self, X, device=None):
        """准备特征张量"""
        if device is None:
            device = self.device
            
        # 确保数据是正确的格式
        if isinstance(X, pd.DataFrame):
            X = X.astype(np.float32).values
        elif not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        else:
            X = X.astype(np.float32)
            
        # 确保是2D
        X = self.converter.ensure_2d(X)
        
        # 转换为张量
        return self.converter.to_tensor(X, device)
    
    def prepare_targets(self, y, device=None):
        """准备目标张量（用于分类）"""
        if device is None:
            device = self.device
            
        # 转换为张量
        y_tensor = self.converter.to_tensor(y, device)
        
        # 确保正确的形状和类型用于分类
        if y_tensor.dim() == 0:
            y_tensor = y_tensor.unsqueeze(0)
            
        return y_tensor.long()
    
    def create_dataset(self, X, y=None, device=None):
        """创建TensorDataset"""
        from torch.utils.data import TensorDataset
        
        if device is None:
            device = self.device
            
        X_tensor = self.prepare_features(X, device)
        
        if y is not None:
            y_tensor = self.prepare_targets(y, device)
            return TensorDataset(X_tensor, y_tensor)
        else:
            return TensorDataset(X_tensor)
    
    def create_dataloader(self, X, y=None, batch_size=32, shuffle=True, device=None):
        """创建DataLoader"""
        from torch.utils.data import DataLoader
        
        dataset = self.create_dataset(X, y, device)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def batch_to_device(self, batch, device=None):
        """将batch移动到指定设备"""
        if device is None:
            device = self.device
            
        if isinstance(batch, (list, tuple)):
            return [item.to(device) if torch.is_tensor(item) else item for item in batch]
        elif torch.is_tensor(batch):
            return batch.to(device)
        else:
            return batch
    
    def validate_tensor_shapes(self, X_tensor, y_tensor=None):
        """验证张量形状"""
        if X_tensor.dim() != 2:
            raise ValueError(f"Features tensor must be 2D, got {X_tensor.dim()}D")
            
        if y_tensor is not None:
            if y_tensor.dim() != 1:
                raise ValueError(f"Target tensor must be 1D, got {y_tensor.dim()}D")
            if X_tensor.size(0) != y_tensor.size(0):
                raise ValueError(f"Batch size mismatch: features {X_tensor.size(0)}, targets {y_tensor.size(0)}")
        
        return True
    