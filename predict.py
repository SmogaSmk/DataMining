from tqdm import tqdm 
import torch 
import numpy as np 
from .utils.tensor_convert import TensorConverter

class ModelPredictor:
  '''
  the predictor designed for models suitable for Pytorch frame
  Has already supported instant prediction
  '''
  
  def __init__(self, model, device = 'cpu'): 
    self.model = model.to(device)
    self.device = device 
  
  def predict_batch(self, input_data):
    '''
    predict for batchs of data
    ''' 
    if not torch.is_tensor(input_data): 
      raise ValueError("Input must be a tensor")
    
    self.model.eval()
    with torch.no_grad(): 
      output = self.model(input_data)
      _, predicted = torch.max(output, 1)
      
    return TensorConverter.to_numpy(predicted)
  
  def predict_proba(self, input_data):
    '''return the probability of predictions'''
    if not torch.is_tensor(input_data): 
      raise ValueError("Input must be a tensor")
    
    self.model.eval()
    with torch.no_grad(): 
      output = self.model(input_data)
      probabilities = torch.softmax(output, dim = 1)
      
    return TensorConverter.to_numpy(probabilities)
  
  def load_weights(self, model_path): 
    self.model.load(torch.load(model_path, map_location=self.device))
    self.model.eval()
    return self
  