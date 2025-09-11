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
            result_data[f'{column}_hour'] = datetime_col.dt.hour
        elif feature == 'day':
            result_data[f'{column}_day'] = datetime_col.dt.day
        elif feature == 'month':
            result_data[f'{column}_month'] = datetime_col.dt.month
        elif feature == 'year':
            result_data[f'{column}_year'] = datetime_col.dt.year
        elif feature == 'dayofweek':
            result_data[f'{column}_dayofweek'] = datetime_col.dt.dayofweek
    
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