import pandas as pd
import pyreadr
import numpy as np

df = pyreadr.read_r('data1_prepared.Rdata')['data1_prepared']

df = df[(df['h_anger'] >= 3.0) | (df['h_anger'] < 2.0)]

df['label'] = 0.0
df.loc[df['h_anger'] >= 3.0, 'label'] = 1.0

df = df[['Text', 'label']]

df = df.groupby(['Text'], as_index=False)['label'].agg(pd.Series.mode)

idxs_unclear = np.array([ type(x) != np.float64 for x in np.array(df['label']) ]).nonzero()[0]

df = df.loc[~df.index.isin(idxs_unclear)]

print(df)

df['Text'].to_csv('all-x.csv', header=None, index=None)
df['label'].to_csv('all-y.csv', header=None, index=None)

print(np.asarray(df['Text'].duplicated()).nonzero()[0].sum())
print(np.unique(np.asarray(df['label']), return_counts=True))