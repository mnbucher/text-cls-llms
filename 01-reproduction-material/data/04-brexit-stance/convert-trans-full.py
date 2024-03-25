import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import deepl
from tqdm import tqdm
import re

# preprocessing: translate non EN sentences with DeepL

# df = pd.read_csv('combined_files_CODED.txt', sep='\t')

# DEEPL_AUTH_KEY = "dfa6b2a3-b2d2-bae4-41a4-1ca7ee0e05fb"
# translator = deepl.Translator(DEEPL_AUTH_KEY)

# for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
	
# 	if row.loc['country'] in ["AUT", "DEU", "FRA"]:
# 		try:
			
# 			if isinstance(row.loc['sentence_before'], str):
# 				result = translator.translate_text(row.loc['sentence_before'], target_lang="EN-GB")
# 				df.loc[idx, 'sentence_before'] = result.text

# 			if isinstance(row.loc['content'], str):
# 				result = translator.translate_text(row.loc['content'], target_lang="EN-GB")
# 				df.loc[idx, 'content'] = result.text

# 			if isinstance(row.loc['sentence_after'], str):
# 				result = translator.translate_text(row.loc['sentence_after'], target_lang="EN-GB")
# 				df.loc[idx, 'sentence_after'] = result.text

# 			df.to_csv('combined_files_CODED_translated.txt', sep='\t')

# 		except Exception as exc:
# 			print(exc)
# 			print(f"could not translate for idx: {idx}")˙˙


df = pd.read_csv('combined_files_CODED_translated.txt', sep='\t')
print(df)

# 1 -> 0
idxs = (df[df['coding_variable_num'] == 1]).index
df.loc[np.array(idxs, dtype=int), 'coding_variable_num'] = 0.0

# 2 -> 1
idxs2 = (df[df['coding_variable_num'] == 2]).index
df.loc[np.array(idxs2, dtype=int), 'coding_variable_num'] = 1.0

# 3 -> 2
idxs2 = (df[df['coding_variable_num'] == 3]).index
df.loc[np.array(idxs2, dtype=int), 'coding_variable_num'] = 2.0

# 4 -> 2
idxs2 = (df[df['coding_variable_num'] == 4]).index
df.loc[np.array(idxs2, dtype=int), 'coding_variable_num'] = 2.0


# k -> 1
idxs = (df[df['coding_variable_k'] == 1]).index
df.loc[np.array(idxs, dtype=int), 'coding_variable_num'] = 1.0

# kk -> 2
idxs2 = (df[df['coding_variable_k'] == 2]).index
df.loc[np.array(idxs2, dtype=int), 'coding_variable_num'] = 1.0

df = df[~df['coding_variable_num'].isna()]

df = df[['doc_id', 'sentence_before', 'content', 'sentence_after', 'coding_variable_num']]

df['text_concat'] = df.apply(lambda x: (x['sentence_before'] if isinstance(x['sentence_before'], str) else "") + " " + (x['content'] if isinstance(x['content'], str) else "") + "" + (x['sentence_after'] if isinstance(x['sentence_after'], str) else ""), axis=1)

df = df.groupby(['text_concat'], as_index=False)['coding_variable_num'].agg(pd.Series.mode)

idxs_unclear = np.array([ type(x) != np.float64 for x in np.array(df['coding_variable_num']) ]).nonzero()[0]
df = df.loc[~df.index.isin(idxs_unclear)]

df['text_concat'] = df['text_concat'].apply(lambda x: x.replace("---", ""))
df['text_concat'] = df['text_concat'].apply(lambda x: re.sub("\s\s+", " ", x))

pd.DataFrame(df['text_concat']).to_csv('all-x.csv', index=None, header=None)
pd.DataFrame(df['coding_variable_num']).to_csv('all-y.csv', index=None, header=None)

# check duplicates and distribution
print(np.asarray(df['text_concat'].duplicated()).nonzero()[0].sum())
print(np.unique(np.asarray(df['coding_variable_num']), return_counts=True))
