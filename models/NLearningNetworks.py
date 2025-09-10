import numpy as np  
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
from abc import ABC, abstractmethod

'''
----------------------------------
To better combined with our homework, 
we build a new file for classic models which 
cannot be well integrated with the Pytorch framework
----------------------------------
'''

class BaseModel(ABC):
  
  @abstractmethod 
  def fit(self, X, y): 
    pass 
  
  @abstractmethod
  def predict(self, X):
    pass
  
  def to(self, device):
    return self 
    
class _KNNCore: 
  def __init__(self, k = 100):
    self.k = k 
    
  def distance(self, X_train, test_point):
    return np.sqrt(np.sum((X_train - test_point) ** 2, axis = 1))
  
  def find_k_nearest(self, X_train, y_train, test_point):
    distances = self.distance(X_train, test_point)
    k_near = min(self.k, len(distances))
    k_indices = np.argpartition(distances, k_near - 1)[:k_near]
    
    return [(distances[i], y_train[i]) for i in k_indices]
  
  def aggregate_reg(self, k_neighbors):
    if len(k_neighbors) == 0: 
      return 0.0
    
    # weighted_avg 
    weights = [1.0 / (dist_i + 1e-8) for dist_i, y_i in k_neighbors]
    weighted_sum = sum(w * label for w, (_, label) in zip(weights, k_neighbors))
    weight_sum = sum(weights)
    
    return weighted_sum / weight_sum
    
  def aggregate_classify(self, k_neighbors): 
    if len(k_neighbors) == 0: 
      return 0
    
    # Voting 
    labels = [label for _, label in k_neighbors]
    
    count_dict = {}
    
    for label in labels: 
      if isinstance(label, np.ndarray): 
        label_key = tuple(label) if label.ndim > 0 else label.item()
      else : 
        label_key = label
      count_dict[label_key] = count_dict.get(label_key, 0) + 1
    return max(count_dict, key = count_dict.get)

class KNN(BaseModel): 
  def __init__(self, k = 100, task = 'classification', n_jobs = -1):
    self.core = _KNNCore(k)
    self.task = task
    self.n_jobs = n_jobs
    
    self.X_train = None 
    self.y_train = None
    self.is_fitted = False
    
    self.feature_names = None
    self.n_features = None 
    self.n_classes = None
    
  def _deTensorfy(self, data):
    
    if torch.is_tensor(data): 
      data = data.detach().cpu().numpy()
    else:
      data = np.array(data)
      
    return data 
    
  def fit(self, X, y):
    
    if torch.is_tensor(X): 
      self.X_train = X.detach().cpu().numpy()
    else: 
      self.X_train = np.array(X)
      
    if torch.is_tensor(y):
      self.y_train = y.detach().cpu().numpy()
    else: 
      self.y_train = np.array(y)
      
    self.n_features = self.X_train.shape[1]
    if self.task_type == 'classification':
      self.n_classes = len(np.unique(self.y_train))

    self.is_fitted = True
    
    # print(f"KNN fitted: {self.X_train.shape[0]} samples, {self.n_features} features")
    if self.task_type == 'classification': 
      print(f"Number of classes: {self.n_classes}")
      
    return self
  
  def predict(self, X):
    if not self.is_fitted:
      raise ValueError("Please fit the model first")
    
    if torch.is_tensor(X):
      X_np = X.detach().cpu().numpy()
    else: 
      X_np = np.array(X)
      
    if X_np.ndim == 1: 
      X_np = X_np.reshape(1, -1)
      return self._predict_single(X_np[0])
    else: 
      return self._predict_batch(X_np[0])
    
  def _predict_single(self, test_point):
    k_neighbors = self.core.find_k_nearest(self.X_train, self.y_train, test_point)
    if self.task == 'classification': 
      return self.core.aggregate_classify(k_neighbors)
    else: 
      return self.core.aggregate_reg(k_neighbors)
    
  def _predict_batch(self, X_test): 
    predictions = Parallel(n_jobs = self.n_jobs) (
      delayed(self._predict_single)(test_point) for test_point in X_test
    )
    
    return np.array(predictions)
  
  def predict_probability(self, X):
    if self.task_type != 'classification': 
      raise ValueError("Only available for classification task")
    
    if torch.is_tensor(self, X): 
      X_np = X.detach().cpu().numpy()
    else : 
      X_np = np.array(X)
    
    if X_np.ndim == 1: 
      X_np = X_np.reshape(1, -1)
      
    probabilities = [] 
    for test_point in X_np: 
      neighbors = self.core.find_k_nearest(self.X_train, self.y_train, test_point)
      labels = [label for _, label in neighbors]
      
      class_probs = np.zeros(self.n_classes)
      for label in labels: 
        class_probs[int(label)] += 1 
      class_probs /= len(labels)
      probabilities.append(class_probs)
      
    return np.array(probabilities)
  
  def save(self, path):

    if not self.is_fitted:
      raise ValueError("Model not fitted")
        
    save_dict = {
      'model_type': 'knn',
      'X_train': self.X_train,
      'y_train': self.y_train,
      'k': self.core.k,
      'task_type': self.task_type,
      'distance_metric': self.core.distance_metric,
      'n_features': self.n_features,
      'n_classes': self.n_classes
    }
        
    np.savez_compressed(path, **save_dict)
    print(f"KNN model saved to {path}")
    
  def load(self, path):

    data = np.load(path, allow_pickle=True)
        
    self.X_train = data['X_train']
    self.y_train = data['y_train']
    self.core.k = int(data['k'])
    self.task_type = str(data['task_type'])
    self.core.distance_metric = str(data['distance_metric'])
    self.n_features = int(data['n_features'])
    self.n_classes = data['n_classes'].item() if data['n_classes'].ndim == 0 else None
    self.is_fitted = True
        
    print(f"KNN model loaded from {path}")
    return self
  
  def __repr__(self):
    status = "fitted" if self.is_fitted else "not fitted"
    return f"KNN(k={self.core.k}, task='{self.task_type}', {status})"
          