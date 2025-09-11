from .models import KNN 
from .models import LinearMultiClassification
from .utils.data_utils import encode_categorical_column, scale_features, encode_target_column
from .utils.tensor_convert import TensorConverter
from train import KNNTrainer
from predict import KNNPredictor
import torch
import pandas as pd
import numpy as np 
from abc import ABC, abstractmethod 
from train import ClassificationTrainer

class BasePipeline(ABC):
  def __init__(self, model, scaler_type = 'standard'):
    self.model = model 
    self.scaler_type = scaler_type
    self.is_fitted = False 

  @abstractmethod
  def preprocess_features(self, X, is_training=True):
    pass 

  @abstractmethod 
  def preprocess_target(self, y, is_training=True): 
    pass 

  @abstractmethod
  def fit(self, X, y, **kwargs): 
    pass 

  @abstractmethod
  def predict(self, X, y):
    pass 

class MLPipeline_NP(BasePipeline):
  def __init__(self, model, scaler_type='StandardScaler'):
    super.__init__(model, scaler_type)

    self.scalers={}
    self.encoders={}
    self.target_encoder=None 
    self.numeric_columns=[]
    self.categorial_columns=[]

  def preprocess_features(self, X, is_training=True):
    X_processed = X.copy()

    if isinstance(X_processed, pd.DataFrame): 
      # identify the column type 
      if is_training: 
        self.numeric_columns = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        self.categorial_columns = X_processed.select_dtypes(include=['object']).columns.tolist()

      # deal with classification label 
      for col in self.categorial_columns:
        if is_training:
          X_processed[col], self.encoders[col] = encode_categorical_column(X_processed, col)
        else: 
          if col in self.encoders: 
            X_processed[col] = self.encoders[col].transform(X_processed[col].astype(str))
          else: 
            raise ValueError(f"No encoder found for column {col}")

      if self.numeric_columns: 
        if is_training: 
          X_processed, self.scalers['feature'] = scale_features(
            X_processed, self.numeric_columns, self.scaler_type
          )
        else: 
          X_processed, _ = scale_features(
            X_processed, self.numeric_columns, 
            self.scaler_type, fit_scaler=self.scalers['features']
          )

    X_numpy = TensorConverter.to_numpy(X_processed)
    if X_numpy.ndim == 1: 
      X_numpy = X_numpy.reshape(1, -1)
    
    return X_numpy
  
  def preprocess_target(self, y, is_training=True):
    if is_training: 
      y_processed, self.target_encoder = encode_target_column(y)
    else: 
      if self.target_encoder is not None: 
        y_processed = self.target_encoder.transform(y)
      else: 
        y_processed = TensorConverter.to_numpy(y)

    return y_processed
  
  def fit(self, X, y): 
    X_processed = self.preprocess_features(X, is_training=True)
    y_processed = self.preprocess_target(y, is_training=True)

    self.model.fit(X_processed, y_processed)
    self.is_fitted=True
    return self 

  def predict(self, X): 
    if not self.is_fitted: 
      raise ValueError("Pipeline must be fitted before making predictions")
    
    X_processed = self.preprocess_features(X, is_training=False)
    predictions = self.model.predict(X_processed)

    if self.target_encoder is not None: 
      predictions = self.target_encoder.inverse_transform(predictions.astype(int))

    return predictions
  
  def predict_proba(self, X): 
    if not hasattr(self.model, 'predict_proba'): 
      raise ValueError("Model doesn't ")
    
    X_processed = self.preprocess_features(X, is_training=False)
    return self.model.predict_proba(X_processed)

class MLPipeline_P(BasePipeline):
  def __init__(self, model, scaler_type='standard', device='cuda'):
    super().__init__(model, scaler_type)

    self.device = device 
    self.model = self.model.to(device)
    self.trainer = None 

  def fit(self, X, y, validation=None, **kwargs):
    X_tensor = self.preprocess_features(X, is_training=True)
    y_tensor = self.preprocess_target(y, is_training=True)

    train_loader = self._create_dataloader(X_tensor, y_tensor, **kwargs)
    val_loader = None 
    if validation: 
      val_loader = self._create_validation_data(validation)
    
    self.trainer = ClassificationTrainer
