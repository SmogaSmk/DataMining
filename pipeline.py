from .models import KNN 
from .models import LinearMultiClassification
from train import KNNTrainer
from predict import KNNPredictor
import torch

class MLPipeline: 
  def __init__(self, model_type, device = 'cpu', enable_save=False, **kwargs): 
    '''
    Pipeline for the whole task from data processing to generate the answer
    : Args : 
      model_type: choose the model to complete the task
      device: usually we use cpu or cuda(cuda can only be used for Learnable model)
       or sth. like apple silicon which support the Pytorch framework
      enable_save: save the model or not
      **kwargs: 
        usually relative to the model itself
        - k: KNN to select the number of neighbors
        - input_size, output_size: the 
    '''
    self.model_type = model_type 
    self.device = device
    self.enable_save = enable_save
    
    if model_type == 'knn': 
      self.model = KNN(**kwargs)
      self.trainer = KNNTrainer()
      self.predictor = KNNPredictor()
    elif model_type == 'linear'
      input_size = kwargs.get('input_size', 'hidden_size')
      output_size = kwargs.get('output_size', 3)
      self.model = LinearMultiClassification(input_size, output_size).to(device)
      # self.trainer = NeuralTrainer()
      # self.predictor = NeuralPredictor()
    
    else: 
      raise ValueError(f"Unsupported model type: {model_type}")
    
  def train(self, train_loader, val_loader = None, **kwargs): 
    return self.trainer.train(self.model, train_loader, val_loader, **kwargs)
  
  def predict(self, test_loader):
    return self.predictor.predict(self.model, test_loader)
  
  def save_model(self, path): 
    '''save the model after training on your computer(optional)'''
    print("IF YOU WANT TO SAVE THE MODEL, PLEASE BE ASSURE enable_save IS True!!!")
    if not self.enable_save: 
      print("Model saving is disabled. Set enable_save=True to enable this feature!")
      return 
    
    if hasattr(self.model, 'save'): 
      self.model.save('path')
    else: 
      torch.save(self.model.state_dict(), path)
      
    print('Model saved to {path}')
    
  def load_model(self, path): 
    '''Load the model from your local space (optional)'''
    if not self.enable_save:
      print("Model loading is disabled. Set enable_save=True to enable this feature.")
      return
        
    if hasattr(self.model, 'load'):
      self.model.load(path)
    else:
      self.model.load_state_dict(torch.load(path, map_location=self.device))
      print(f"Model loaded from {path}")