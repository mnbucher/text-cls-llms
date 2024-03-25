import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 11285 samples
df = pd.read_csv('./3SU.csv')

# 4148 samples
df = df[df['relevance'] == 'yes']

# set positive to 1.0
df.loc[df['positivity'] == 'positive', 'positivity'] = 1.0

# set neutral, mixed and 'not sure' to 0.0 (neutral)
df.loc[df['positivity'].isin(['neutral', 'mixed', 'not sure']), 'positivity'] = 0.0

# set negative to -1.0
df.loc[df['positivity'] == 'negative', 'positivity'] = -1.0

# majority voting on label
df = df.groupby(['sentenceid', 'headline', 'text'], as_index=False).agg(label=('positivity', pd.Series.mode))

# set values with multiple modes to neutral (will be filtered out later)
df.loc[df['label'].apply(lambda x: not isinstance(x, float)), 'label'] = 0.0

# only take samples with positive (1.0) or negative (-1.0) label
df = df[df['label'] != 0.0]

# set negative label from -1.0 to 0.0
df.loc[df['label'] == -1.0, 'label'] = 0.0

print("total # of samples:", len(df))

print(np.unique(df['label'], return_counts=True))

df['text'].to_csv('all-x.csv', header=False, index=False)
df['label'].to_csv('all-y.csv', header=False, index=False)