from tqdm import tqdm 
import torch 
import numpy as np 

class KNNPredictor: 
  def __init__(self):
    pass 
  
  def predict(self, model, test_loader): 
    '''
    Use the model to predict the res
    :Args:
      model: the model we use to predict the results, usually the KNN model
      test_loader: prepare for test data to examine the model
    ''' 
    
    if not model.is_fitted: 
      raise ValueError("Model not fitted")
    
    predictions = []
    
    print("KNN prediction ...")
    for batch_x, _ in tqdm(test_loader, desc = "Predicting"): 
      batch_predictions = model.predict(batch_x)
      if not isinstance(batch_predictions, (list, np.ndarray)): 
        batch_predictions = [batch_predictions]
      predictions.extend(batch_predictions)
      
    return np.array(predictions)
      
    
    
