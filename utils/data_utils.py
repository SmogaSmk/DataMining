import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

# 全局变量存储处理状态
_label_mapping = {}
_reverse_label_mapping = {}
_scaler = None
_categorical_encoders = {}
_numeric_columns = []
_categorical_columns = []

def create_label_mapping(labels):
  """创建标签映射
    
  Args:
    labels: 标签列表或Series
        
  Returns:
    dict: 标签到索引的映射字典
  """
  global _label_mapping, _reverse_label_mapping
    
  if isinstance(labels, pd.Series):
    unique_labels = labels.unique()
  else:
    unique_labels = np.unique(labels)
    
  unique_labels = sorted(unique_labels)
  _label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
  _reverse_label_mapping = {idx: label for label, idx in _label_mapping.items()}
    
  print(f"创建了 {len(unique_labels)} 个类别的标签映射:")
  for original, encoded in _label_mapping.items():
    print(f"  {original} -> {encoded}")
    
  return _label_mapping

def encode_labels(labels):
  """编码标签
    
  Args:
    labels: 原始标签列表或Series
        
  Returns:
    numpy.ndarray: 编码后的标签数组
  """
  global _label_mapping
    
  if not _label_mapping:
    create_label_mapping(labels)
    
  if isinstance(labels, pd.Series):
    encoded = labels.map(_label_mapping)
    if encoded.isna().any():
      unknown_labels = labels[encoded.isna()].unique()
      raise ValueError(f"发现未知标签: {unknown_labels}")
    return encoded.values
  else:
    encoded = np.array([_label_mapping.get(label, -1) for label in labels])
    if (encoded == -1).any():
      unknown_indices = np.where(encoded == -1)[0]
      unknown_labels = np.array(labels)[unknown_indices]
      raise ValueError(f"发现未知标签: {np.unique(unknown_labels)}")
    return encoded

def decode_labels(encoded_labels):

  global _reverse_label_mapping
    
  if not _reverse_label_mapping:
    raise ValueError("标签映射未创建，请先调用 create_label_mapping")
    
  # 处理 torch.Tensor
  if hasattr(encoded_labels, 'cpu'):
    encoded_labels = encoded_labels.cpu().numpy()
  elif not isinstance(encoded_labels, np.ndarray):
    encoded_labels = np.array(encoded_labels)
    
  decoded = np.array([_reverse_label_mapping.get(int(label), f"Unknown_{int(label)}")
                    for label in encoded_labels])
  return decoded

def preprocess_features(features, fit=True, scaler_type='standard', categorical_columns=None):
  """预处理特征数据
    
  Args:
    features: 特征数据 (DataFrame或array)
    fit: 是否需要拟合处理器
    scaler_type: 缩放器类型 ('standard', 'minmax', 'robust')
    categorical_columns: 分类列名列表
        
  Returns:
    numpy.ndarray: 处理后的特征数据
  """
  global _scaler, _categorical_encoders, _numeric_columns, _categorical_columns
    
    # 确保输入是DataFrame
  if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)
    
  result_data = features.copy()
    
  if fit:
    # 自动检测列类型
    if categorical_columns is None:
            categorical_columns = result_data.select_dtypes(include=['object']).columns.tolist()
        
    _categorical_columns = categorical_columns
    _numeric_columns = result_data.select_dtypes(include=[np.number]).columns.tolist()
        
    print(f"检测到 {len(_numeric_columns)} 个数值列: {_numeric_columns}")
    print(f"检测到 {len(_categorical_columns)} 个分类列: {_categorical_columns}")
        
    # 处理分类变量
    _categorical_encoders = {}
    for col in _categorical_columns:
      if col in result_data.columns:
        encoder = LabelEncoder()
        result_data[col] = encoder.fit_transform(result_data[col].astype(str))
        _categorical_encoders[col] = encoder
        print(f"编码分类列: {col}")
        
        # 处理数值变量缩放
    if _numeric_columns:
      if scaler_type == 'standard':
        _scaler = StandardScaler()
      elif scaler_type == 'minmax':
        _scaler = MinMaxScaler()
      elif scaler_type == 'robust':
        _scaler = RobustScaler()
      else:
        raise ValueError("只支持 'standard', 'minmax', 'robust' 缩放器")
            
      result_data[_numeric_columns] = _scaler.fit_transform(result_data[_numeric_columns])
      print(f"缩放数值列: {_numeric_columns}")
    
  else:
    # 使用已拟合的处理器
    # 处理分类变量
    for col in _categorical_columns:
      if col in result_data.columns and col in _categorical_encoders:
        encoder = _categorical_encoders[col]
        result_data[col] = encoder.transform(result_data[col].astype(str))
        
    # 处理数值变量缩放
    if _numeric_columns and _scaler is not None:
      result_data[_numeric_columns] = _scaler.transform(result_data[_numeric_columns])
    
    # 转换为float32数组
    return result_data.astype(np.float32).values

def prepare_data(data, target_column, feature_columns=None, test_size=0.2, random_state=42, scaler_type='standard'):

  from sklearn.model_selection import train_test_split
    
  if feature_columns is None:
    feature_columns = [col for col in data.columns if col != target_column]
    
  X = data[feature_columns]
  y = data[target_column]
    
  # 分割数据
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
  )
    
  # 预处理特征
  X_train_processed = preprocess_features(X_train, fit=True, scaler_type=scaler_type)
  X_test_processed = preprocess_features(X_test, fit=False)
    
  # 编码标签
  y_train_encoded = encode_labels(y_train)
  y_test_encoded = encode_labels(y_test)
  
  return X_train_processed, X_test_processed, y_train_encoded, y_test_encoded

def get_preprocessing_info():

  global _label_mapping, _reverse_label_mapping, _scaler, _categorical_encoders
  global _numeric_columns, _categorical_columns
    
  return {
    'label_mapping': _label_mapping.copy(),
    'reverse_label_mapping': _reverse_label_mapping.copy(),
    'scaler': _scaler,
    'categorical_encoders': _categorical_encoders.copy(),
    'numeric_columns': _numeric_columns.copy(),
    'categorical_columns': _categorical_columns.copy()
  }

def load_preprocessing_info(info):

  global _label_mapping, _reverse_label_mapping, _scaler, _categorical_encoders
  global _numeric_columns, _categorical_columns
    
  _label_mapping = info.get('label_mapping', {})
  _reverse_label_mapping = info.get('reverse_label_mapping', {})
  _scaler = info.get('scaler', None)
  _categorical_encoders = info.get('categorical_encoders', {})
  _numeric_columns = info.get('numeric_columns', [])
  _categorical_columns = info.get('categorical_columns', [])

def reset_preprocessing_state():
  
  global _label_mapping, _reverse_label_mapping, _scaler, _categorical_encoders
  global _numeric_columns, _categorical_columns
    
  _label_mapping = {}
  _reverse_label_mapping = {}
  _scaler = None
  _categorical_encoders = {}
  _numeric_columns = []
  _categorical_columns = []
  print("预处理状态已重置")

def get_num_classes():
  """获取类别数量"""
  return len(_label_mapping)

def get_class_labels():
  """获取原始类别标签列表"""
  if not _label_mapping:
    raise ValueError("标签映射未创建")
  return sorted(_label_mapping.keys())

# 保留原有的高级函数用于向后兼容
def extract_datetime_features(data, column, features_to_extract=['hour']):
  """从单个时间列提取时间特征"""
  result_data = data.copy()
  datetime_col = pd.to_datetime(data[column])
    
  for feature in features_to_extract:
    if feature == 'hour':
      result_data[f'{column}'] = datetime_col.dt.hour
    elif feature == 'day':
      result_data[f'{column}'] = datetime_col.dt.day
    elif feature == 'month':
      result_data[f'{column}'] = datetime_col.dt.month
    elif feature == 'year':
      result_data[f'{column}'] = datetime_col.dt.year
    elif feature == 'dayofweek':
      result_data[f'{column}'] = datetime_col.dt.dayofweek
    
  return result_data

def auto_preprocess_data(X_train, X_test=None, y_train=None, y_test=None, 
                        scaler_type='standard', handle_missing=True):
  
  reset_preprocessing_state()
    
  # 预处理特征
  X_train_processed = preprocess_features(X_train, fit=True, scaler_type=scaler_type)
  X_test_processed = None
  if X_test is not None:
    X_test_processed = preprocess_features(X_test, fit=False)
    
  # 预处理标签
  y_train_processed = None
  y_test_processed = None
  if y_train is not None:
    y_train_processed = encode_labels(y_train)
  if y_test is not None:
    y_test_processed = encode_labels(y_test)
    
  result = {
    'X_train_processed': X_train_processed,
    'X_test_processed': X_test_processed,
    'y_train_processed': y_train_processed,
    'y_test_processed': y_test_processed,
    'preprocessing_info': get_preprocessing_info()
  }
    
  print("✅ 数据预处理完成！")
  return result