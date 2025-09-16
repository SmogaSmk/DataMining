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