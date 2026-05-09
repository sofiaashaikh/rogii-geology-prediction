import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_and_engineer(df):
    df = df.copy()
    
    df['GR'] = df['GR'].interpolate(method='linear', limit_direction='both')
    
    df['Delta_X'] = df['X'].diff().fillna(0)
    df['Delta_Y'] = df['Y'].diff().fillna(0)
    df['Delta_Z'] = df['Z'].diff().fillna(0)
    df['GR_Roll_Mean_5'] = df['GR'].rolling(window=5, min_periods=1).mean()
    df['GR_Roll_Std_5'] = df['GR'].rolling(window=5, min_periods=1).std().fillna(0)
    df['GR_Roll_Mean_10'] = df['GR'].rolling(window=10, min_periods=1).mean()
    
    features = [
        'MD', 'X', 'Y', 'Z', 'GR', 
        'Delta_X', 'Delta_Y', 'Delta_Z', 
        'GR_Roll_Mean_5', 'GR_Roll_Std_5', 'GR_Roll_Mean_10'
    ]
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df, features