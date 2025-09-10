import numpy as np 
import torch 

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
  def to_tensor(data, device = 'gpu'): 
    '''Transform python origin, numpy, pandas or other data structure to tensor'''
    if torch.is_tensor(data): 
      return data.to(device)
    else: 
      return torch.from_numpy(np.array(data)).to(device)
    
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
    