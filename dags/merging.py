import sklearn.pipeline as pipe
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime as dt

df3['date_time'] = pd.to_datetime(df3['date_time'], format = '%d.%m.%Y %H:%M')
df4 = pd.merge(df,df3,how="outer")
df4=df4.sort_values(by='date_time')

df4.ffill(inplace=True)
df4.fillna(method='bfill',inplace=True)
df4 = df4.reset_index(drop=True)
