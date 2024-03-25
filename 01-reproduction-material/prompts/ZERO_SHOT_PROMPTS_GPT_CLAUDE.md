# Zero-Shot Prompts for ChatGPT

After experimenting through the API with a few prompts, we fine-tuned our final prompt to maximize the likelihood that the model only returns one label from the given list of available class labels. As ChatGPT is a probabilistic model, it's stochasticity leads to a variance in its generated output. By controlling the temperature parameter and decreasing its value, we can decrease the variance of the generated output. However, the model still generates other content than just one of the given output labels. In the following, we present the prompts we provided to the ChatGPT API, where \<Text\> marks the text placeholder for the current text sample.


## Sentiment Analysis on The New York Times Articles covering US Economy

Prompt:

```
You have been assigned the task of zero-shot text classification for sentiment analysis. Your objective is to classify a given text snippet into one of several possible class labels, based on the sentiment expressed in the text. Your output should consist of a single class label that best matches the sentiment expressed in the text. Your output should consist of a single class label that best matches the given text. Choose ONLY from the given class labels below and ONLY output the label without any other characters.

Text: <Text>

Labels: 'Negative Sentiment', 'Positive Sentiment'

Answer:
```


## Stance Classification on Tweets about Kavanaugh Election

Prompt:

```
You have been assigned the task of zero-shot text classification for stance classification. Your objective is to classify a given text snippet into one of several possible class labels, based on the attitudinal stance towards the given text. Your output should consist of a single class label that best matches the stance expressed in the text. Your output should consist of a single class label that best matches the given text. Choose ONLY from the given class labels below and ONLY output the label without any other characters.

Text: <Text>

Labels: 'negative attitudinal stance towards', 'positive attitudinal stance towards'

Answer:
```


## Emotion Classification on Political Texts in German

Prompt:

```
You have been assigned the task of zero-shot text classification for emotion classification. Your objective is to classify a given text snippet into one of several possible class labels, based on the anger level in the given text. Your output should consist of a single class label that best matches the anger expressed in the text. Choose ONLY from the given class labels below and ONLY output the label without any other characters.

Text: <Text>

Labels: 'Angry', 'Non-Angry'

Answer:
```


## Multi-Class Stance Classification on Brexit Corpus

Prompt:

```
You have been assigned the task of zero-shot text classification for political texts on attitudinal stance towards Brexit and leave demands related to the European Union (EU). Your objective is to classify a given text snippet into one of several possible class labels, based on the stance towards Brexit and general leave demands in the given text. Your output should consist of a single class label that best matches the content expressed in the text. Choose ONLY from the given class labels below and ONLY output the label without any other characters.

Text: <Text>

Labels: 'Neutral towards Leave demands', 'Pro-Leave demands', 'Very Pro-Leave demands'

Answer:
```