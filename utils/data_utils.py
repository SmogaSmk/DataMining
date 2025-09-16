import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
  """数据预处理器 - 封装状态管理和核心预处理逻辑"""
  
  def __init__(self, scaler_type='standard'):
    """
    Args:
      scaler_type: 缩放器类型 ('standard', 'minmax', 'robust')
    """
    self.scaler_type = scaler_type
    self.reset()
  
  def reset(self):
    """重置预处理状态"""
    self.label_mapping = {}
    self.reverse_label_mapping = {}
    self.scaler = None
    self.categorical_encoders = {}
    self.numeric_columns = []
    self.categorical_columns = []
  
  def create_label_mapping(self, labels):
    """创建标签映射"""
    if isinstance(labels, pd.Series):
      unique_labels = labels.unique()
    else:
      unique_labels = np.unique(labels)
        
    unique_labels = sorted(unique_labels)
    self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    self.reverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}
    
    print(f"创建了 {len(unique_labels)} 个类别的标签映射:")
    for original, encoded in self.label_mapping.items():
      print(f"  {original} -> {encoded}")
        
    return self.label_mapping
  
  def encode_labels(self, labels):
    """编码标签"""
    if not self.label_mapping:
      self.create_label_mapping(labels)
        
    if isinstance(labels, pd.Series):
      encoded = labels.map(self.label_mapping)
      if encoded.isna().any():
        unknown_labels = labels[encoded.isna()].unique()
        raise ValueError(f"发现未知标签: {unknown_labels}")
      return encoded.values
    else:
      encoded = np.array([self.label_mapping.get(label, -1) for label in labels])
      if (encoded == -1).any():
        unknown_indices = np.where(encoded == -1)[0]
        unknown_labels = np.array(labels)[unknown_indices]
        raise ValueError(f"发现未知标签: {np.unique(unknown_labels)}")
      return encoded
  
  def decode_labels(self, encoded_labels):
    """解码标签"""
    if not self.reverse_label_mapping:
      raise ValueError("标签映射未创建，请先调用 create_label_mapping")
        
    # 处理 torch.Tensor
    if hasattr(encoded_labels, 'cpu'):
      encoded_labels = encoded_labels.cpu().numpy()
    elif not isinstance(encoded_labels, np.ndarray):
      encoded_labels = np.array(encoded_labels)
        
    decoded = np.array([self.reverse_label_mapping.get(int(label), f"Unknown_{int(label)}")
                      for label in encoded_labels])
    return decoded
  
  def preprocess_features(self, features, fit=True, categorical_columns=None):
    """预处理特征数据"""
    # 确保输入是DataFrame
    if not isinstance(features, pd.DataFrame):
      features = pd.DataFrame(features)
    
    result_data = features.copy()
    
    if fit:
      # 自动检测列类型
      if categorical_columns is None:
        categorical_columns = result_data.select_dtypes(include=['object']).columns.tolist()
      
      self.categorical_columns = categorical_columns
      self.numeric_columns = result_data.select_dtypes(include=[np.number]).columns.tolist()
      
      print(f"检测到 {len(self.numeric_columns)} 个数值列: {self.numeric_columns}")
      print(f"检测到 {len(self.categorical_columns)} 个分类列: {self.categorical_columns}")
      
      # 处理分类变量
      self.categorical_encoders = {}
      for col in self.categorical_columns:
        if col in result_data.columns:
          encoder = LabelEncoder()
          result_data[col] = encoder.fit_transform(result_data[col].astype(str))
          self.categorical_encoders[col] = encoder
          print(f"编码分类列: {col}")
      
      # 处理数值变量缩放
      if self.numeric_columns:
        if self.scaler_type == 'standard':
          self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
          self.scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
          self.scaler = RobustScaler()
        else:
          raise ValueError("只支持 'standard', 'minmax', 'robust' 缩放器")
        
        result_data[self.numeric_columns] = self.scaler.fit_transform(result_data[self.numeric_columns])
        print(f"缩放数值列: {self.numeric_columns}")
    
    else:
      # 使用已拟合的处理器
      # 处理分类变量
      for col in self.categorical_columns:
        if col in result_data.columns and col in self.categorical_encoders:
          encoder = self.categorical_encoders[col]
          result_data[col] = encoder.transform(result_data[col].astype(str))
      
      # 处理数值变量缩放
      if self.numeric_columns and self.scaler is not None:
        result_data[self.numeric_columns] = self.scaler.transform(result_data[self.numeric_columns])
    
    return result_data.astype(np.float32).values
  
  def get_num_classes(self):
    """获取类别数量"""
    return len(self.label_mapping)
  
  def get_class_labels(self):
    """获取原始类别标签列表"""
    if not self.label_mapping:
      raise ValueError("标签映射未创建")
    return sorted(self.label_mapping.keys())
  
  def save_state(self):
    """保存预处理状态"""
    return {
      'label_mapping': self.label_mapping.copy(),
      'reverse_label_mapping': self.reverse_label_mapping.copy(),
      'scaler': self.scaler,
      'categorical_encoders': self.categorical_encoders.copy(),
      'numeric_columns': self.numeric_columns.copy(),
      'categorical_columns': self.categorical_columns.copy(),
      'scaler_type': self.scaler_type
    }
  
  def load_state(self, state):
    """加载预处理状态"""
    self.label_mapping = state.get('label_mapping', {})
    self.reverse_label_mapping = state.get('reverse_label_mapping', {})
    self.scaler = state.get('scaler', None)
    self.categorical_encoders = state.get('categorical_encoders', {})
    self.numeric_columns = state.get('numeric_columns', [])
    self.categorical_columns = state.get('categorical_columns', [])
    self.scaler_type = state.get('scaler_type', 'standard')


# 保持简单的函数式接口用于快速使用
def prepare_data(data, target_column, feature_columns=None, test_size=0.2, 
                 random_state=42, scaler_type='standard'):
  """
  快速数据准备函数 - 适合简单场景的一站式处理
  
  Args:
    data: 原始数据DataFrame
    target_column: 目标列名
    feature_columns: 特征列名列表，None则自动使用所有其他列
    test_size: 测试集比例
    random_state: 随机种子
    scaler_type: 缩放器类型
    
  Returns:
    tuple: (X_train, X_test, y_train, y_test, preprocessor)
  """
  if feature_columns is None:
    feature_columns = [col for col in data.columns if col != target_column]
    
  X = data[feature_columns]
  y = data[target_column]
  
  # 分割数据
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
  )
  
  # 创建预处理器并处理数据
  preprocessor = DataPreprocessor(scaler_type=scaler_type)
  
  # 预处理特征
  X_train_processed = preprocessor.preprocess_features(X_train, fit=True)
  X_test_processed = preprocessor.preprocess_features(X_test, fit=False)
  
  # 编码标签
  y_train_encoded = preprocessor.encode_labels(y_train)
  y_test_encoded = preprocessor.encode_labels(y_test)
  
  print("✅ 数据预处理完成！")
  return X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, preprocessor


def auto_preprocess_data(X_train, X_test=None, y_train=None, y_test=None, 
                         scaler_type='standard', handle_missing=True):
  """
  自动预处理数据函数 - 兼容原有接口
  
  Returns:
    dict: 包含处理后数据和预处理器的字典
  """
  preprocessor = DataPreprocessor(scaler_type=scaler_type)
  
  # 预处理特征
  X_train_processed = preprocessor.preprocess_features(X_train, fit=True)
  X_test_processed = None
  if X_test is not None:
    X_test_processed = preprocessor.preprocess_features(X_test, fit=False)
    
  # 预处理标签
  y_train_processed = None
  y_test_processed = None
  if y_train is not None:
    y_train_processed = preprocessor.encode_labels(y_train)
  if y_test is not None:
    y_test_processed = preprocessor.encode_labels(y_test)
    
  result = {
    'X_train_processed': X_train_processed,
    'X_test_processed': X_test_processed,
    'y_train_processed': y_train_processed,
    'y_test_processed': y_test_processed,
    'preprocessor': preprocessor,  # 返回预处理器实例而不是状态字典
    'preprocessing_info': preprocessor.save_state()  # 保持向后兼容
  }
    
  print("✅ 数据预处理完成！")
  return result


# 实用工具函数保持函数式
def extract_datetime_features(data, column, features_to_extract=['hour']):
  """从单个时间列提取时间特征"""
  result_data = data.copy()
  datetime_col = pd.to_datetime(data[column])
    
  for feature in features_to_extract:
    if feature == 'hour':
      result_data[f'{column}_hour'] = datetime_col.dt.hour
    elif feature == 'day':
      result_data[f'{column}_day'] = datetime_col.dt.day
    elif feature == 'month':
      result_data[f'{column}_month'] = datetime_col.dt.month
    elif feature == 'year':
      result_data[f'{column}_year'] = datetime_col.dt.year
    elif feature == 'dayofweek':
      result_data[f'{column}_dayofweek'] = datetime_col.dt.dayofweek
  
  # 删除原始列
  result_data = result_data.drop(columns=[column])
  return result_data