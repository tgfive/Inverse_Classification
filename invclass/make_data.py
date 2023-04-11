import pandas as pd
import numpy as np

#PATH = ''
PATH = '../brazil_data/'
OUT = 'A036_weather'
FILE = 'central_west.csv'
DESC = 'columns_description.csv'

STATION = 'A036'

SAMPLE = True
SAMPLE_SIZE = 1000

print('Loading data...')

df = pd.read_csv(PATH+FILE)
col_desc = pd.read_csv(PATH+DESC)

print('Mapping labels...')

df.drop(columns=['index'], inplace=True)

mapping = pd.Series(col_desc['abbreviation'].values, index=df.columns).to_dict()
df.rename(columns=mapping, inplace=True)

df = df.loc[df['inme'] == STATION]
df.drop(columns=['reg','prov','wsnm','inme'], inplace=True)

print('Creating time values...')

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

date_time = df.pop('mdct')

timestamp_s = date_time.map(pd.Timestamp.timestamp)

day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

print('Approximating missing points...')

df.replace([-9999,-99990], np.nan, inplace=True)
nandf = df.isnull().astype('int')
for col in df.columns:
    df[col].fillna(value=df[col].mean(), inplace=True)

print('Finalizing dataset...')

df['wd_rad'] = df['wdct'] * np.pi / 180

df['Wx'] = df['wdsp'] * np.cos(df['wd_rad'])
df['Wy'] = df['wdsp'] * np.sin(df['wd_rad'])
nandf['Wx'] = np.max([nandf['wdsp'], nandf['wdct']])
nandf['Wy'] = nandf['Wx']

df['Gx'] = df['gust'] * np.cos(df['wd_rad'])
df['Gy'] = df['gust'] * np.sin(df['wd_rad'])
nandf['Gx'] = np.max([nandf['gust'], nandf['wdct']])
nandf['Gy'] = nandf['Gx']

df.drop(columns=['wd_rad','wdsp','wdct','gust'], inplace=True)
nandf.drop(columns=['wdsp','wdct','gust'], inplace=True)

df['target'] = df.sum(axis=1)
nandf['target'] = nandf.sum(axis=1)

noise = np.random.normal(0,1,(df.shape[0],3))
df[['lat', 'lon', 'elvt']] = df[['lat', 'lon', 'elvt']] + noise

df = df[['lat', 'lon', 'elvt', 'Day sin', 'Day cos', 'Year sin', 'Year cos',
    'smax', 'smin', 'tmax', 'tmin', 'dmax', 'dmin', 'hmax', 'hmin', 'Gx', 'Gy',
    'prcp', 'stp', 'gbrd', 'temp', 'dewp', 'hmdy', 'Wx', 'Wy',
    'target']]
nandf = nandf[['lat', 'lon', 'elvt', 'Day sin', 'Day cos', 'Year sin', 'Year cos',
    'smax', 'smin', 'tmax', 'tmin', 'dmax', 'dmin', 'hmax', 'hmin', 'Gx', 'Gy',
    'prcp', 'stp', 'gbrd', 'temp', 'dewp', 'hmdy', 'Wx', 'Wy',
    'target']]


print('Writing to file...')

if SAMPLE:
    inds = list(np.arange(0,SAMPLE_SIZE))
    df = df.iloc[inds]
    nandf = nandf.iloc[inds]

df.index = np.arange(1, len(df.index)+1)
df.to_csv(PATH+OUT+'.csv', index=True)

nandf.index = np.arange(1, len(nandf.index)+1)
nandf.to_csv(PATH+OUT+'_nan.csv', index=True)
