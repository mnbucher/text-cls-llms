import pandas as pd
import re

def clean_row(x):
	x = re.sub("RT ", "", x)
	x = re.sub("https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)", "", x)
	tweet_exp = '@\w{2,15}'
	tweet_handle = re.compile(tweet_exp)
	
	while tweet_handle.search(x):
		x = re.sub(tweet_exp, "", x)

	x = re.sub("\s\s+", " ", x)
	x = re.sub("^\s", "", x)
	x = re.sub(r"\"", '', x)
	
	return x

df = pd.read_csv('kavanaugh_tweets_groundtruth.csv', index_col=None)[['text', 'stance']]
df['text'] = df['text'].apply(lambda x: clean_row(x))

df = df.groupby(['text'], as_index=False)['stance'].agg(pd.Series.mode).reset_index()

df = df.iloc[1:, :]

all_x = df['text']
all_x.to_csv('all-x.csv', header=None, index=None)

df['stance'].to_csv('all-y.csv', header=None, index=None)