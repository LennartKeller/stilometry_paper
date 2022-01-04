#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import datetime
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import sent_tokenize
from pathlib import Path
from dateutil.parser import parse as parse_date
from lxml import etree
from tqdm.auto import tqdm
from tehut.normalisation import cab
from flair.models import SequenceTagger, MultiTagger
from flair.data import Sentence
from nltk import sent_tokenize


# In[ ]:


import inspect
src = inspect.getsource(cab)
print(src)


# In[ ]:


orig_df = pd.read_csv('../data/rom_real_dataset_full.csv')


# In[ ]:


MODELS = ['de-pos', 'de-ner-large',]
tagger = MultiTagger.load(MODELS)


# In[ ]:


for idx, row in orig_df.iterrows():
    orig_text = row['text']
    n_megabytes = len(orig_text.encode('utf-8')) // 1000000
    if n_megabytes >= 1:
        print(idx)
        print(n_megabytes)
        print(row['author'], row['title'])
        normed_text = cab(orig_text)
        sentences = [Sentence(s) for s in sent_tokenize(normed_text, language='german')]
        tagged_sentences = []
        for sentence in tqdm(sentences):
            tagger.predict(sentence)
            tagged_sentences.append(sentence.to_tagged_string())
        tagged_text = ' '.join(tagged_sentences)
        orig_df.at[idx, 'normed_text'] = normed_text
        orig_df.at[idx, 'tagged_text'] = tagged_text


# In[ ]:


author_parts = []
for pnd in orig_df.author_pnd.unique():
    author_df = orig_df.query('author_pnd == @pnd')
    if len(author_df) > 15:
        author_df = author_df.sample(15)
    author_parts.append(author_df)
sampled_df = pd.concat(author_parts, axis=0).reset_index(drop=True)


# In[ ]:


orig_df.to_csv('../data/rom_real_dataset_full_repaired.csv', index=False)
sample_df.to_csv('../data/rom_real_dataset_repaired.csv', index=False)

