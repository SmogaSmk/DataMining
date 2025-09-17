from .models import KNN 
from .models import LinearMultiClassification
from .utils.data_utils import DataPreprocessor  # 改为使用新的类
from .utils.tensor_convert import TensorConverter, TensorDataProcessor
from .predict import ModelPredictor
from .train import ClassificationTrainer
import torch
import pandas as pd
import numpy as np 
from abc import ABC, abstractmethod 

from torch.utils.data import TensorDataset, DataLoader

class BasePipeline(ABC):
  def __init__(self, model, scaler_type='standard'):
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
    # 创建数据预处理器实例
    self.preprocessor = DataPreprocessor(scaler_type=scaler_type)

  def preprocess_features(self, X, is_training=True):
    X_processed = self.preprocessor.preprocess_features(X, fit=is_training)
    
    # 转换为numpy数组
    X_numpy = TensorConverter.to_numpy(X_processed)
    if X_numpy.ndim == 1: 
      X_numpy = X_numpy.reshape(1, -1)
    
    return X_numpy
  
  def preprocess_target(self, y, is_training=True):
    if is_training:
      y_processed = self.preprocessor.encode_labels(y)
    else:
      y_processed = self.preprocessor.encode_labels(y)
    
    return TensorConverter.to_numpy(y_processed)
  
  def fit(self, X, y): 
    X_processed = self.preprocess_features(X, is_training=True)
    y_processed = self.preprocess_target(y, is_training=True)

    self.model.fit(X_processed, y_processed)
    self.is_fitted = True
    return self 

  def predict(self, X): 
    if not self.is_fitted: 
      raise ValueError("Pipeline must be fitted before making predictions")
    
    X_processed = self.preprocess_features(X, is_training=False)
    predictions = self.model.predict(X_processed)

    # 使用预处理器进行反转换
    return self.preprocessor.decode_labels(predictions)
  
  def predict_proba(self, X): 
    if not hasattr(self.model, 'predict_proba'): 
      raise ValueError("Model doesn't support predict_proba")
    
    X_processed = self.preprocess_features(X, is_training=False)
    return self.model.predict_proba(X_processed)

class MLPipeline_P(BasePipeline):
  """PyTorch 模型管道"""
  def __init__(self, model, scaler_type='standard', device='cuda'):
    super().__init__(model, scaler_type)

    self.device = device 
    self.model = self.model.to(device)
    self.trainer = None 
    
    # 创建数据预处理器实例
    self.preprocessor = DataPreprocessor(scaler_type=scaler_type)
    self.tensor_processor = TensorDataProcessor(device)
    
    self.optimizer = None
    self.criterion = None
    self.predictor = ModelPredictor(self.model, device)
    
  def preprocess_features(self, X, is_training=True, categorical_columns=None):
    """预处理特征数据"""
    X_processed = self.preprocessor.preprocess_features(
      X, fit=is_training, categorical_columns=categorical_columns
    )
      
    # 转换为张量
    return self.tensor_processor.prepare_features(X_processed, self.device)

  def preprocess_target(self, y, is_training=True):
    """预处理目标变量"""
    if is_training:
      y_encoded = self.preprocessor.encode_labels(y)
    else:
      y_encoded = self.preprocessor.encode_labels(y)
    
    # 转换为张量
    return self.tensor_processor.prepare_targets(y_encoded, self.device)
  
  def fit(self, X, y, epochs=100, batch_size=32, 
          categorical_columns=None, validation=None, **kwargs):
    """训练模型"""
    print(f"Pipeline device: {self.device}")
    print(f"Model device: {next(self.model.parameters()).device}")
    
    # 确保输入数据是 float32 类型
    if isinstance(X, pd.DataFrame):
      X = X.astype({col: np.float32 for col in X.select_dtypes(include=[np.number]).columns})
    else:
      X = np.array(X, dtype=np.float32)
    
    # 预处理数据
    X_tensor = self.preprocess_features(X, is_training=True, categorical_columns=categorical_columns)
    y_tensor = self.preprocess_target(y, is_training=True)
    
    print(f"X_tensor device: {X_tensor.device}")
    print(f"y_tensor device: {y_tensor.device}")
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None 
    if validation is not None:
      X_val, y_val = validation
      # 确保验证数据也是 float32 类型
      if isinstance(X_val, pd.DataFrame):
        X_val = X_val.astype({col: np.float32 for col in X_val.select_dtypes(include=[np.number]).columns})
      else:
        X_val = np.array(X_val, dtype=np.float32)
          
      X_val_tensor = self.preprocess_features(X_val, is_training=False, categorical_columns=categorical_columns)
      y_val_tensor = self.preprocess_target(y_val, is_training=False)
      
      val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
      val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
          
    self.trainer = ClassificationTrainer(
      model=self.model, 
      device=self.device, 
      **kwargs
    )
    
    # 开始训练
    train_losses, val_accuracies = self.trainer.train(
      train_loader=train_loader, 
      val_loader=val_loader,
      epochs=epochs
    )
    
    self.is_fitted = True
    return train_losses, val_accuracies
  
  def predict(self, X, categorical_columns=None, return_original_labels=True): 
    """预测接口"""
    if not self.is_fitted:
      raise ValueError("Model must be fitted before prediction")
    
    X_processed = self.preprocess_features(X, is_training=False, categorical_columns=categorical_columns)
    encoded_predictions = self.predictor.predict_batch(X_processed)
    
    if return_original_labels:
      return self.preprocessor.decode_labels(encoded_predictions)
    else:
      return encoded_predictions
    
  def get_class_labels(self): 
    """获取类别标签"""
    return self.preprocessor.get_class_labels()
  
  def get_label_mapping(self): 
    """获取标签映射"""
    return self.preprocessor.label_mapping
  
  def predict_proba(self, X, categorical_columns=None): 
    if not self.is_fitted: 
      raise ValueError("Model must be fitted before prediction")
    
    X_processed = self.preprocess_features(X, is_training=False, categorical_columns=categorical_columns)
    return self.predictor.predict_proba(X_processed)
  
  def save_model(self, model_path):
    if not self.is_fitted: 
      raise ValueError("Model must be fitted before saving")
    
    torch.save(self.model.state_dict(), model_path)
    
    # 保存预处理信息
    preprocessing_info = self.preprocessor.save_state()
    
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
        
      # 加载到预处理器
      self.preprocessor.load_state(preprocessing_info)
      
      print(f"Model loaded from: {model_path}")
      print(f"Preprocessing info loaded from: {preprocessing_path}")
      print(f"Available classes: {self.preprocessor.get_class_labels()}")
            
    except FileNotFoundError:
      print(f"Warning: Preprocessing file {preprocessing_path} not found. "
            "Label mapping may not work correctly.")
        
    self.is_fitted = True 
    return self