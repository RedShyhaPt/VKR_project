import sklearn.pipeline as pipe
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime as dt
from deep_translator import GoogleTranslator
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer, AutoModel

df2 = pd.read_csv("yand_news.csv",sep=";")
df2_2 = pd.read_csv("eco.csv",sep=";")
df2_3 = pd.read_csv("poly.csv",sep=";")


df2['content_'], df2['aug_c'] =' ',' '
for i in df2.index:
  df2['content_'][i] = GoogleTranslator(source='auto', target='german').translate(df2['news'][i])
  df2['aug_c'][i] = GoogleTranslator(source='auto', target='russian').translate(df2['content_'][i])

df2_2['content_'], df2_2['aug_a'] =' ',' '
for i in df2_2.index:
  df2_2['content_'][i] = GoogleTranslator(source='auto', target='german').translate(df2_2['news'][i])
  df2_2['aug_a'][i] = GoogleTranslator(source='auto', target='russian').translate(df2_2['content_'][i])

df2_3['content_'], df2_3['aug_b'] =' ',' '
for i in df2_3.index:
  df2_3['content_'][i] = GoogleTranslator(source='auto', target='german').translate(df2_3['news'][i])
  df2_3['aug_b'][i] = GoogleTranslator(source='auto', target='russian').translate(df2_3['content_'][i])

df2 = df2.drop('content_', axis=1)
df2_2 = df2_2.drop('content_', axis=1)
df2_3 = df2_3.drop('content_', axis=1)

#переводим текст в нижний регистр
df2[['news','aug_c']] = df2[['news','aug_c']].apply(lambda x: x.astype(str).str.lower())
df2_2[['news','aug_a']] = df2_2[['news','aug_a']].apply(lambda x: x.astype(str).str.lower())
df2_3[['news','aug_b']] = df2_3[['news','aug_b']].apply(lambda x: x.astype(str).str.lower())

df2['aug_c'] = df2['aug_c'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
df2_2['aug_a'] = df2_2['aug_a'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
df2_3['aug_b'] = df2_3['aug_b'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))

stop = stopwords.words('russian')
df2['aug_c'] = df2['aug_c'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df2_2['aug_a'] = df2_2['aug_a'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df2_3['aug_b'] = df2_3['aug_b'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#переводим текст в нижний регистр
df2['news'] = df2['news'].apply(lambda x: x.lower())
df2_2['news'] = df2_2['news'].apply(lambda x: x.lower())
df2_3['news'] = df2_3['news'].apply(lambda x: x.lower())


df2['news'] = df2['news'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
df2_2['news'] = df2_2['news'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
df2_3['news'] = df2_3['news'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))

stop = stopwords.words('russian')
df2['news_d'] = df2['news'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df2_2['news_d'] = df2_2['news'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df2_3['news_d'] = df2_3['news'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df2['news_c'] = df2['news_d']
df2_2['news_a'] = df2_2['news_d']
df2_3['news_b'] = df2_3['news_d']
df2 = df2.drop(['news','news_d'], axis=1)
df2_2 = df2_2.drop(['news','news_d'], axis=1)
df2_3 = df2_3.drop(['news','news_d'], axis=1)

df3_concated = pd.merge(df2, df2_2,how="outer")
df3_concated=df3_concated.sort_values(by='date_time')
df3 = pd.merge(df3_concated,df2_3,how="outer")
df3=df3.sort_values(by='date_time')
df3.ffill(inplace=True)
df3.fillna(method='bfill',inplace=True)

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
# model.cuda()  # uncomment it if you have a GPU

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].numpy()


df3['bert_text_a'], df3['bert_text_b'], df3['bert_text_c'] = ' ', ' ', ' '
df3['bert_aug_a'], df3['bert_aug_b'], df3['bert_aug_c'] = = ' ', ' ', ' '
for i in df3.index:
  df3['bert_text_a'][i] = embed_bert_cls(df3['news_a'][i], model, tokenizer)
  df3['bert_text_b'][i] = embed_bert_cls(df3['news_b'][i], model, tokenizer)
  df3['bert_text_c'][i] = embed_bert_cls(df3['news_c'][i], model, tokenizer)

  df3['bert_aug_a'][i] = embed_bert_cls(df3['aug_a'][i], model, tokenizer)
  df3['bert_aug_b'][i] = embed_bert_cls(df3['aug_b'][i], model, tokenizer)
  df3['bert_aug_c'][i] = embed_bert_cls(df3['aug_c'][i], model, tokenizer) 
