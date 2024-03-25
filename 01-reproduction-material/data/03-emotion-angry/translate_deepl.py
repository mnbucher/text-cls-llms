import pandas
import http.client
import requests
from tqdm import tqdm
import pandas as pd


DEEPL_PRO_HOST = "https://api.deepl.com"

DEEPL_URL_USAGE = "/v2/usage?auth_key="
DEEPL_URL_TRANSLATE = "/v2/translate"

DEEPL_AUTH_KEY = "dfa6b2a3-b2d2-bae4-41a4-1ca7ee0e05fb"


df = pandas.read_csv('./all-x.csv', header=None, index_col=None)

texts = list(df.iloc[:, 0])

# get usage
r = requests.get(DEEPL_PRO_HOST + DEEPL_URL_USAGE + DEEPL_AUTH_KEY)
resp_usage = r.json()
usage_remaining = resp_usage.get("character_limit") - resp_usage.get("character_count")
print("Characters remaining : {}".format(usage_remaining))

translations = []

# Translate row by row
for i in tqdm(range(0, len(texts))):

	text = texts[i]

	try:
		if not (isinstance(text, float) and math.isnan(text)) and text != 0 and text != '0':
			
			payload = {
				"target_lang": "EN",
				"auth_key": DEEPL_AUTH_KEY,
				"source_lang": "DE",
				"text": text
			}

			r = requests.post(DEEPL_PRO_HOST + DEEPL_URL_TRANSLATE, params=payload) #+ DEEPL_AUTH_KEY + "&source_lang=" + source_lang + "&text=" + text)

			resp_transl = r.json()
			translation = resp_transl.get("translations")[0].get("text")

	except Exception as exc:
		print("exception!")
		print(exc)
		translation = ""

	translations.append(translation)


pd.DataFrame(translations).to_csv('all-x-translated.csv', index=None, header=None)