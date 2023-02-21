import pandas as pd
import numpy as np
import random
import os

print(os.getcwd())

OUT = 'brazil_data/brazil_weather'
FILE = 'brazil_data/central_west.csv'
DESC = 'brazil_data/columns_description.csv'

SAMPLE = True
SAMPLE_SIZE = 650

df = pd.read_csv(FILE)
col_desc = pd.read_csv(DESC)

df.drop(columns=['index'], inplace=True)

mapping = pd.Series(col_desc['abbreviation'].values, index=df.columns).to_dict()
df.rename(columns=mapping, inplace=True)

df['DATE'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['TIME'] = pd.to_datetime(df['hr'], format='%H:%M')

df.drop(columns=['date','hr'], inplace=True)

df['DAY'] = df['DATE'].dt.day
df['MONTH'] = df['DATE'].dt.month
df['YEAR'] = df['DATE'].dt.year
df['HOUR'] = df['TIME'].dt.hour
df['MINUTE'] = df['TIME'].dt.minute
df['SECOND'] = df['TIME'].dt.second

df['mdct'] = pd.to_datetime(df[['DAY','MONTH','YEAR','HOUR','MINUTE','SECOND']])

df.drop(columns=['DATE','TIME','DAY','MONTH','YEAR','HOUR','MINUTE','SECOND'], inplace=True)

df.sort_values(by='mdct', inplace=True)
df.reset_index(drop=True, inplace=True)

df.drop(columns=['reg','prov','wsnm','inme'], inplace=True)

df.replace([-9999,-99990], np.nan, inplace=True)
for col in df.columns:
    df[col].fillna(value=df[col].mean(), inplace=True)

df['wd_rad'] = df['wdct'] * np.pi / 180

df['Wx'] = df['wdsp'] * np.cos(df['wd_rad'])
df['Wy'] = df['wdsp'] * np.sin(df['wd_rad'])

df['Gx'] = df['gust'] * np.cos(df['wd_rad'])
df['Gy'] = df['gust'] * np.sin(df['wd_rad'])

df.drop(columns=['wd_rad','wdsp', 'wdct','gust'], inplace=True)

date_time = df.pop('mdct')

timestamp_s = date_time.map(pd.Timestamp.timestamp)

day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

df['target'] = df.sum(axis=1)

#df = (df - df.min()) / (df.max() - df.min())

df = df[['lat', 'lon', 'elvt', 'Day sin', 'Day cos', 'Year sin', 'Year cos',
    'smax', 'smin', 'tmax', 'tmin', 'dmax', 'dmin', 'hmax', 'hmin', 'Gx', 'Gy',
    'prcp', 'stp', 'gbrd', 'temp', 'dewp', 'hmdy', 'Wx', 'Wy',
    'target']]

if SAMPLE:
    inds = [random.randint(0,len(df.index)) for iter in range(SAMPLE_SIZE)]
    df = df.iloc[inds]

df.index = np.arange(1, len(df.index)+1)

df.to_csv(OUT+'.csv', index=True)