from .models import KNN 
from .models import LinearMultiClassification
from .utils.data_utils import encode_categorical_column, scale_features, encode_target_column
from .utils.tensor_convert import TensorConverter
from .predict import ModelPredictor
from .train import ClassificationTrainer
import torch
import pandas as pd
import numpy as np 
from abc import ABC, abstractmethod 

from torch.utils.data import TensorDataset, DataLoader

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
  def __init__(self, model, scaler_type='standard'):
    super().__init__(model, scaler_type)

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
            self.scaler_type, fit_scaler=self.scalers['feature']
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
    
    # y_processed = TensorConverter.to_numpy(y_processed)
    # print(type(y_processed))
    return y_processed
  
  def fit(self, X, y): 
    X_processed = self.preprocess_features(X, is_training=True)
    y_processed = self.preprocess_target(y, is_training=True)

    y_processed = TensorConverter.to_numpy(y_processed)
    # print(type(y_processed))
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
    self.converter = TensorConverter()
    
    self.feature_scaler = {}
    self.categorical_encoders = {}
    self.target_scaler = None
    self.label_encoder = None
    
    # Label Projection
    self.label_mapping = {}  
    self.inverse_label_mapping = {}
    self.num_classes = None
    self.original_labels = None
    
    self.optimizer = None
    self.criterion = None
    self.predictor = ModelPredictor(self.model, device)
    
  def _create_label_mapping(self, y): 
    if isinstance(y, pd.Series): 
      unique_labels = y.unique()
    else: 
      unique_labels = np.unique(y)
      
    unique_labels = sorted(unique_labels)
    self.original_labels = unique_labels 
    self.num_classes = len(unique_labels)
    
    '''create bi-mapping'''
    self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    self.inverse_label_mapping = {idx: label for idx, label in enumerate(unique_labels)}
    
    print(f"Created label mapping for {self.num_classes} classes: ")
    for original, encoded in self.label_mapping.items(): 
      print(f" {original} -> {encoded}")
  
  def _encode_labels(self, y): 
    if isinstance(y, pd.Series): 
      encoded = y.map(self.label_mapping)
      
      if encoded.isna().any(): 
        unknown_labels = y[encoded.isna()].unique()
        raise ValueError(f"Unknown labels found during prediction: {unknown_labels}")
      return encoded.values
    else: 
      encoded = np.array([self.label_mapping.get(label, -1) for label in y])
      if (encoded == -1).any(): 
        unknown_indicies = np.where(encoded == -1)[0] 
        unknown_labels = np.array(y)[unknown_indicies]
        raise ValueError(f"Unknown labels found during prediction: {np.unique(unknown_labels)}")
      return encoded
    
  def _decode_labels(self, encoded_labels): 
    if isinstance(encoded_labels, torch.Tensor): 
      encoded_labels = TensorConverter.to_numpy(encoded_labels)
    decoded = np.array([self.inverse_label_mapping.get(int(label), f"Unknown_{int(label)}")
                        for label in encoded_labels])
    return decoded 
  
  def preprocess_features(self, X, is_training=True, 
                          categorical_columns = None):
    '''For torch framework, we need to switch it to Tensor'''
    if not isinstance(X, pd.DataFrame): 
      X = pd.DataFrame(X)

    result_data = X.copy() 
    
    if categorical_columns: 
      for col in categorical_columns: 
        if col in result_data.columns: 
          if is_training: 
            encoded_series, encoder = encode_categorical_column(result_data, col)
            self.categorical_encoders[col] = encoder 
            result_data[col] = encoded_series 
          else: 
            if col in self.categorical_encoders:
              encoder = self.categorical_encoders[col]
              encoded_values = encoder.transform(result_data[col].astype(str))
              result_data[col] = encoded_values 
    
    numeric_columns = result_data.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_columns: 
      if is_training: 
        result_data, scaler = scale_features(
          result_data, numeric_columns, self.scaler_type
        ) 
        self.feature_scaler = scaler
      else: 
        if self.feature_scaler is not None: 
          result_data, _ = scale_features(
            result_data, numeric_columns, self.scaler_type, 
            fit_scaler=self.feature_scaler
          )
          
    return self.converter.to_tensor(result_data.values, self.device)

  def preprocess_target(self, y, is_training=True):
    '''Process target variable for PyTorch training'''
    if is_training:
      self._create_label_mapping(y)
      y_encoded = self._encode_labels(y)
    else: 
      if self.label_mapping is None : 
        raise ValueError("Model must be fitted before prediction")
      y_encoded = self._encode_labels(y)
    
    # Convert to tensor
    y_tensor = self.converter.to_tensor(y_encoded, self.device)
    
    # Ensure proper shape and type for classification
    if y_tensor.dim() == 0:
      y_tensor = y_tensor.unsqueeze(0)
    
    return y_tensor.long()
  
  def fit(self, X, y, epochs=100, batch_size=32, 
          categorical_columns = None, validation=None, **kwargs):
    
    print(f"Pipeline device: {self.device}")
    print(f"Model device: {next(self.model.parameters()).device}")
    
    X_tensor = self.preprocess_features(X, is_training=True, 
                                        categorical_columns=categorical_columns)
    y_tensor = self.preprocess_target(y, is_training=True)
    
    print(f"X_tensor device: {X_tensor.device}")
    print(f"y_tensor device: {y_tensor.device}")
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None 
    if validation is not None : 
      X_val, y_val = validation
      X_val_tensor = self.preprocess_features(X_val, is_training=False, 
                                              categorical_columns=categorical_columns)
      y_val_tensor = self.preprocess_target(y_val, is_training=False)
      val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
      val_loader = DataLoader(val_dataset, batch_size=batch_size)
          
    self.trainer = ClassificationTrainer(
      model = self.model, 
      device = self.device, 
      **kwargs
    )
    
    train_losses, val_accuracies = self.trainer.train(
      train_loader=train_loader, 
      val_loader=val_loader,
      epochs = epochs
    )
    
    self.is_fitted = True
    return train_losses, val_accuracies
  
  def predict(self, X, categorical_columns=None, return_original_labels=True): 
    '''
    predict interface(Pipeline config the data, and Predictor execute the prediction)
    '''
    if not self.is_fitted:
      raise ValueError("Model must be fitted before prediction")
    
    X_processed = self.preprocess_features(X,
                                           is_training=False, 
                                           categorical_columns=categorical_columns)
    
    encoded_predictions = self.predictor.predict_batch(X_processed)
    
    if return_original_labels: 
      return self._decode_labels(encoded_predictions)
    else:
      return encoded_predictions
    
  def get_class_labels(self): 
    if self.original_labels is None: 
      raise ValueError("Model must be fitted first")
    return self.original_labels.copy()
  
  def get_label_mapping(self): 
    if not self.label_mapping: 
      raise ValueError("Model must be fitted first")
    return self.label_mapping.copy()
  
  def predict_proba(self, X, categorical_columns=None): 
    if not self.is_fitted: 
      raise ValueError("Model must be fitted before prediction")
    
    X_processed = self.preprocess_features(X, is_training=False, 
                    categorical_columns=categorical_columns)
    
    return self.predictor.predict_proba(X_processed)
  
  def save_model(self, model_path):
    if not self.is_fitted: 
      raise ValueError("Model must be fitted before saving")
    
    torch.save(self.model.state_dict(), model_path)
    
    preprocessing_info = {
      'feature_scaler': self.feature_scaler, 
      'categorical_encoders': self.categorical_encoders,
      'label_mapping': self.label_mapping,
      'inverse_label_mapping': self.inverse_label_mapping,
      'original_labels': self.original_labels,
      'num_classes': self.num_classes,
      'scaler_type': self.scaler_type
    }
    
    preprocessing_path = model_path.replace('.pth', '_preprocessing.pkl')
    import pickle
    with open(preprocessing_path, 'wb') as f:
      pickle.dump(preprocessing_info, f)
    print(f"Model saved to: {model_path}")
    print(f"Preprocessing info saved to: {preprocessing_path}")
        
    return self

  def load_model(self, model_path): 
    self.predictor.load_weights(model_path)
    # 加载预处理信息
    preprocessing_path = model_path.replace('.pth', '_preprocessing.pkl')
    try:
      import pickle
      with open(preprocessing_path, 'rb') as f:
        preprocessing_info = pickle.load(f)
            
        self.feature_scaler = preprocessing_info.get('feature_scaler', {})
        self.categorical_encoders = preprocessing_info.get('categorical_encoders', {})
        self.label_mapping = preprocessing_info.get('label_mapping', {})
        self.inverse_label_mapping = preprocessing_info.get('inverse_label_mapping', {})
        self.original_labels = preprocessing_info.get('original_labels', None)
        self.num_classes = preprocessing_info.get('num_classes', None)
        self.scaler_type = preprocessing_info.get('scaler_type', 'standard')
            
        print(f"Model loaded from: {model_path}")
        print(f"Preprocessing info loaded from: {preprocessing_path}")
        print(f"Available classes: {self.original_labels}")
            
    except FileNotFoundError:
      print(f"Warning: Preprocessing file {preprocessing_path} not found. "
            "Label mapping may not work correctly.")
        
    self.is_fitted=True 
    return self   
