import numpy as np
import torch 
from .tensor_convert import TensorConverter

class ModelEvaluator: 
  
  @staticmethod 
  def accuracy(preds, tars):
    '''calculate the classification accuracy'''
    preds = TensorConverter.to_numpy(preds)
    tars = TensorConverter.to_numpy(tars)
    return np.mean(preds == tars) * 100
  
  @staticmethod 
  def MSE(preds, tars):
    '''calculate the mean square error'''
    preds = TensorConverter.to_numpy(preds)
    tars = TensorConverter.to_numpy(tars)  
    return np.mean((preds - tars) ** 2)
  
  @staticmethod 
  def calculate_r2_score(preds, tars): 
    '''calculate the R^2 score'''
    preds = TensorConverter.to_numpy(preds)
    tars = TensorConverter.to_numpy(tars)
      
    ss_res = np.sum((tars - preds) ** 2)
    ss_tot = np.sum((tars - np.mean(tars)) ** 2)
    
    return 1 - (ss_res / ss_tot)
  
  @staticmethod
  def evaluate_classification_model(model, data_loader, device = 'cpu'):
    '''evaluate the classification model'''
    model.eval()
    all_preds = [] 
    all_tars = [] 
    
    with torch.no_grad(): 
      for data, target in data_loader : 
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_tars.extend(target.cpu().numpy())
        
    accuracy = ModelEvaluator.accuracy(all_preds, all_tars)
    return accuracy
  
  @staticmethod 
  def evaluate_regression_model(model, data_loader, device = 'cpu'): 
    '''evaluate the regression model'''
    model.eval()
    all_preds = []
    all_tars = [] 
    
    with torch.no_grad():
      for data, target in data_loader: 
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        all_preds.extend(output.cpu().numpy())
        all_tars.extend(target.cpu().numpy())
        
    mse = ModelEvaluator.MSE(all_preds, all_tars)
    r2 = ModelEvaluator.calculate_r2_score(all_preds, all_tars)
    
    return {'mse: ': mse, 'r2: ': r2}
  
  @ staticmethod
  def evaluate_NL_model(model, test_loader, task_type='classification'):
    '''evaluate Non Learning and non parameterized models'''
    all_preds = []
    all_tars = []
        
    for batch_x, batch_y in test_loader:
      preds = model.predict(batch_x)
            
      preds = TensorConverter.to_numpy(preds)
      batch_y = TensorConverter.to_numpy(batch_y)
                
      all_preds.extend(preds)
      all_tars.extend(batch_y)
        
    all_preds = np.array(all_preds)
    all_tars = np.array(all_tars)
        
    if task_type == 'classification' or (hasattr(model, 'task_type') and model.task_type == 'classification'):
      accuracy = ModelEvaluator.accuracy(all_preds, all_tars)
      return {'accuracy': accuracy}
    else:  # regression
      mse = ModelEvaluator.mse(all_preds, all_tars)
      r2 = ModelEvaluator.calculate_r2_score(all_preds, all_tars)
      return {'mse': mse, 'r2': r2}