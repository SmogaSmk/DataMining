import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

def extract_datetime_features(data, column, features_to_extract=['hour']):
  """
  从单个时间列提取时间特征
  :param data: DataFrame
  :param column: 时间列名
  :param features_to_extract: 要提取的特征列表 ['hour', 'day', 'month', 'year', 'dayofweek']
  :return: 包含新特征的DataFrame
  """
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

def encode_categorical_column(data, column):
  """
  对单个类别列进行标签编码
  :param data: DataFrame
  :param column: 列名
  :return: (编码后的Series, 编码器)
  """
  encoder = LabelEncoder()
  encoded_values = encoder.fit_transform(data[column].astype(str))
  return pd.Series(encoded_values, index=data.index, name=column), encoder

def encode_target_column(target_series):
  """
  对目标列进行编码(如果是类别的话)
  :param target_series: 目标列
  :return: (编码后的数组, 编码器或None)
  """
  if target_series.dtype == 'object':
    encoder = LabelEncoder()
    encoded_target = encoder.fit_transform(target_series)
    return encoded_target, encoder
  return target_series.values, None

def auto_preprocess_data(X_train, X_test=None, y_train=None, y_test=None, 
                        scaler_type='standard', handle_missing=True):
  """
  自动预处理数据，包括分类变量编码、数值变量缩放等
  
  Args:
    X_train: 训练特征数据
    X_test: 测试特征数据（可选）
    y_train: 训练目标数据（可选）
    y_test: 测试目标数据（可选）
    scaler_type: 缩放器类型 ('standard', 'minmax', 'robust')
    handle_missing: 是否处理缺失值
    
  Returns:
    dict: 包含处理后的数据和编码器的字典
  """
  result = {
    'X_train_processed': None,
    'X_test_processed': None,
    'y_train_processed': None,
    'y_test_processed': None,
    'categorical_encoders': {},
    'feature_scaler': None,
    'target_encoder': None,
    'numeric_columns': [],
    'categorical_columns': []
  }
  
  # 确保输入是DataFrame
  if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train)
  if X_test is not None and not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test)
  
  # 识别列类型
  result['numeric_columns'] = X_train.select_dtypes(include=[np.number]).columns.tolist()
  result['categorical_columns'] = X_train.select_dtypes(include=['object']).columns.tolist()
  
  print(f"检测到 {len(result['numeric_columns'])} 个数值列: {result['numeric_columns']}")
  print(f"检测到 {len(result['categorical_columns'])} 个分类列: {result['categorical_columns']}")
  
  # 复制数据
  X_train_proc = X_train.copy()
  X_test_proc = X_test.copy() if X_test is not None else None
  
  # 处理分类变量
  for col in result['categorical_columns']:
    print(f"正在编码分类列: {col}")
    
    # 合并训练和测试数据来确保编码一致性
    if X_test is not None:
      combined = pd.concat([X_train_proc[col].astype(str), X_test_proc[col].astype(str)])
    else:
      combined = X_train_proc[col].astype(str)
    
    # 创建编码器并拟合
    encoder = LabelEncoder()
    encoder.fit(combined)
    result['categorical_encoders'][col] = encoder
    
    # 应用编码
    X_train_proc[col] = encoder.transform(X_train_proc[col].astype(str))
    if X_test is not None:
      X_test_proc[col] = encoder.transform(X_test_proc[col].astype(str))
  
  # 处理数值变量缩放
  if result['numeric_columns']:
    print(f"正在缩放数值列: {result['numeric_columns']}")
    X_train_proc, result['feature_scaler'] = scale_features(
      X_train_proc, result['numeric_columns'], scaler_type
    )
    if X_test is not None:
      X_test_proc, _ = scale_features(
        X_test_proc, result['numeric_columns'], scaler_type, 
        fit_scaler=result['feature_scaler']
      )
  
  # 处理目标变量
  if y_train is not None:
    result['y_train_processed'], result['target_encoder'] = encode_target_column(y_train)
    if y_test is not None:
      if result['target_encoder'] is not None:
        result['y_test_processed'] = result['target_encoder'].transform(y_test)
      else:
        result['y_test_processed'] = y_test.values if hasattr(y_test, 'values') else y_test
  
  result['X_train_processed'] = X_train_proc
  result['X_test_processed'] = X_test_proc
  
  print("✅ 数据预处理完成！")
  return result

def scale_features(data, columns, scaler_type = 'standard', fit_scaler=None):
  '''
  for the networks are sensitive to dimension and distance 
  :Args: 
    data: usually pandas DataFrame, ndarray, Series
    columns: column to be scaled, force to declare
    scale_type: scaler's type suits different model
    fit_scaler: used for test data
  '''
  if scaler_type == 'standard': 
    scaler = StandardScaler() 
  elif scaler_type == 'minmax': 
    scaler = MinMaxScaler()
  elif scaler_type == 'robust': 
    scaler = RobustScaler()
  else: 
    raise ValueError("Only support 'standard', 'minmax', or 'robust' scaler at present")
  
  if isinstance(data, pd.DataFrame): 
    if columns is None or len(columns) == 0: 
      raise ValueError("for DataFrame, columns must be specified")
    missing_cols = set(columns) - set(data.columns)
    if missing_cols:
      raise ValueError(f"Columns {missing_cols} not found in DataFrame")
    
    result_data = data.copy()
    if fit_scaler is not None:
      scaler = fit_scaler
      scaled_values = scaler.transform(data[columns])
    else:
      scaled_values = scaler.fit_transform(data[columns])

    result_data[columns] = scaled_values
    return result_data, scaler
  
  elif isinstance(data, pd.Series):
    if fit_scaler is not None:
      scaler = fit_scaler
      scaled_values = scaler.transform(data.values.reshape(-1, 1)).flatten()
    else:
      scaled_values = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    result_series = pd.Series(scaled_values, index=data.index, name=data.name)
    return result_series, scaler
  
  elif isinstance(data, np.ndarray): 
    if data.ndim == 1:
      data_2d = data.reshape(-1, 1)
      is_1d = True
    else:
      data_2d = data
      is_1d = False
      
    if fit_scaler is not None: 
      scaler = fit_scaler
      scaled_values = scaler.transform(data_2d)
    else: 
      scaled_values = scaler.fit_transform(data_2d)
      
    if is_1d: 
      scaled_values = scaled_values.flatten() 
    
    return scaled_values, scaler 
  
  else: 
    raise ValueError("Only dataFrame, Series, or ndarray is supported now")