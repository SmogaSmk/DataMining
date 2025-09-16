import numpy as np  
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
from abc import ABC, abstractmethod
from ..utils.tensor_convert import TensorConverter

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
    '''
    find the neighbors who are the k th closer than any others
    :Args:
      X_train: the train data and points
      y_train: the train data's label
      test_point: the point to find k th neighbors from all training data
    '''
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
      count_dict[label] = count_dict.get(label, 0) + 1

    return max(count_dict, key = count_dict.get)

class KNN(BaseModel): 
  def __init__(self, k = 10, task = 'classification', n_jobs = -1):
    self.core = _KNNCore(k)
    self.task = task
    self.n_jobs = n_jobs
    
    self.X_train = None 
    self.y_train = None
    self.is_fitted = False
    
    self.n_features = None 
    self.n_classes = None
    
  def _validate_input(self, X, y = None): 
    if not isinstance(X, np.ndarray): 
      raise TypeError("X must be numpy array")
    
    if X.ndim != 2: 
      raise ValueError("X must be 2D array ")
    
    if y is not None: 
      if not isinstance(y, np.ndarray): 
        raise TypeError("y must be numpy array")
      if len(X) != len(y): 
        raise ValueError("X and y must share same number of samples")

  def fit(self, X, y):
    
    self._validate_input(X, y)
    self.X_train = X.copy()
    self.y_train = y.copy()
    
    self.n_features = self.X_train.shape[1]
    if self.task == 'classification':
      self.n_classes = len(np.unique(self.y_train))

    self.is_fitted = True
    
    # print(f"KNN fitted: {self.X_train.shape[0]} samples, {self.n_features} features")
    if self.task == 'classification': 
      print(f"Number of classes: {self.n_classes}")
      
    return self
  
  def predict(self, X):
    if not self.is_fitted:
      raise ValueError("Please fit the model first")
    
    self._validate_input(X)
      
    if X.shape[1] != self.n_features:
      raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
    
    return self._predict_batch(X)   
   
  def _predict_single(self, test_point):
    k_neighbors = self.core.find_k_nearest(self.X_train, self.y_train, test_point)
    
    if self.task == 'classification': 
      return self.core.aggregate_classify(k_neighbors)
    else: 
      return self.core.aggregate_reg(k_neighbors)
    
  def _predict_batch(self, X_test): 
    
    if self.n_jobs == 1: 
      predictions = [self._predict_single(test_point) for test_point in X_test]
    
    predictions = Parallel(n_jobs = self.n_jobs) (
      delayed(self._predict_single)(test_point) for test_point in X_test
    )
    
    return np.array(predictions)
  
  def predict_probability(self, X):
    if self.task != 'classification': 
      raise ValueError("Only available for classification task")
    
    if not self.is_fitted:
      raise ValueError("Model must be fitted before prediction")
      
    self._validate_input(X)
    
    probabilities = [] 
    for test_point in X: 
      neighbors = self.core.find_k_nearest(self.X_train, self.y_train, test_point)
      labels = [label for _, label in neighbors]
      
      class_counts = np.bincount(labels, minlength=self.n_classes)
      class_probs = class_counts / len(labels)
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
      'task': self.task,
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
    self.task = str(data['task'])
    self.n_features = int(data['n_features'])
    self.n_classes = data['n_classes'].item() if data['n_classes'].ndim == 0 else None
    self.is_fitted = True
        
    print(f"KNN model loaded from {path}")
    return self
  
  def __repr__(self):
    status = "fitted" if self.is_fitted else "not fitted"
    return f"KNN(k={self.core.k}, task='{self.task}', {status})"
