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
  
  def set_model(self, model): 
    '''set the model for instant prediction'''
    self.model = model.to(self.device)
    self.model.eval()
    return self 
    
  def load_model_offline(self, model_class, model_params, model_path) :
    '''
    load the model's weight that is already trained from folders
    '''
    self.model = model_class(**model_params).to(self.device)
    self.model.load_state_dict(torch.load(model_path, map_location = self.device))
    self.model.eval()
    
  def load_weights(self, model_path): 
    if self.model is None: 
      raise ValueError(
        '''No model has been set. Please use
        set_model() first or load_model_from_file()
      ''')
    self.model.load_state_dict(torch.load(model_path, map_location = self.device))
    self.model.eval()
    return self
  
  def _check_model(self): 
    if self.model is None: 
      raise ValueError("No model available, set a model using set_model() or load_model_from_file()")
    
  def predict_single(self, input_data):
    '''
    predict for one sample
    '''
    input_tensor = TensorConverter.to_tensor(input_data, self.device)
    input_tensor = TensorConverter.ensure_2d(input_tensor)
    
    with torch.no_grad(): 
      output = self.model(input_tensor)
      _, predicted = torch.max(output, 1)
      
    return predicted.cpu().item()
  
  def predict_batch(self, input_data):
    '''
    predict for batchs of data
    ''' 
    input_tensor = TensorConverter.to_tensor(input_data, self.device)
    
    with torch.no_grad(): 
      output = self.model(input_tensor)
      _, predicted = torch.max(output, 1)
    
    return TensorConverter.to_numpy(predicted)
  
  def predict_proba(self, input_data):
    '''return the probability of predictions'''
    self._check_model()
    
    input_tensor = TensorConverter.to_tensor(input_data, self.device)
    input_tensor = TensorConverter.ensure_2d(input_tensor)
    
    with torch.no_grad(): 
      output = self.model(input_tensor)
      probabilities = torch.softmax(output, dim = 1)
      
    return TensorConverter.to_numpy(probabilities)
  
# ==================== Usage ====================
'''
1. Instant predict (predict using the model have already been trained): 
# predict after training
trainer = ClassificationTrainer(model, device='cuda')
train_losses, val_accuracies = trainer.train(train_loader, val_loader)

# directly use trained model to predict
predictor = ModelPredictor(trainer.model, device='cuda')
prediction = predictor.predict_single([1.2, 3.4, 2.1, 0.8])

# functions call in a chain
predictor = ModelPredictor(device='cuda').set_model(trainer.model)
prediction = predictor.predict_single([1.2, 3.4, 2.1, 0.8])

2. predict with the model load from disk:
# load through model's file path
predictor = ModelPredictor().load_model_from_file(
    LinearMultiClassification, 
    {'input_size': 4, 'output_size': 3}, 
    'best_model.pth'
)

# set the model first then load its weights
model = LinearMultiClassification(input_size=4, output_size=3)
predictor = ModelPredictor(model).load_weights('best_model.pth')

3. change your equipments
predictor = ModelPredictor(device='cuda')

# predict after training immediately
predictor.set_model(trained_model)
real_time_pred = predictor.predict_single(test_sample)

# load the best model has ever trained
predictor.load_weights('best_model.pth')
final_pred = predictor.predict_single(test_sample)

'''