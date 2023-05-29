# подключить необходимые бибилотеки
import sklearn.pipeline as pipe
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime as dt

df = pd.read_csv("YNDX_230112_230330.csv",sep=";")

df['price'] = (df['<OPEN>']+df['<HIGH>']+df['<LOW>']+df['<CLOSE>'])/4
df['total'] = df['price']*df['<VOL>']
df['date_time'] = pd.to_datetime(df['<DATE>']+' '+df['<TIME>'], format = '%d/%m/%y %H:%M')
df['weekday'] = df['date_time'].dt.weekday
df['month'] = df['date_time'].dt.month
df['hour'] = df['date_time'].dt.hour
df['minute'] = df['date_time'].dt.minute
#df['year'] = df['date_time'].dt.year
df['day'] = df['date_time'].dt.day

df = df.drop('<PER>', axis=1)
df = df.drop('<TICKER>', axis=1)
df = df.drop('<DATE>', axis=1)
df = df.drop('<TIME>', axis=1)
df = df.drop('<OPEN>', axis=1)
df = df.drop('<HIGH>', axis=1)
df = df.drop('<LOW>', axis=1)
df = df.drop('<CLOSE>', axis=1)
df = df.drop('<VOL>', axis=1)
