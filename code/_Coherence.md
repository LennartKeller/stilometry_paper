---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lib import tokenize_lemmatized
from tqdm.auto import tqdm
from nltk import sent_tokenize
tqdm.pandas()
```

```python
corpus = pd.read_csv("../data/rom_real_dataset_final.csv")
corpus.shape
```

```python
indef_articles = ["ein", "eine", "einer", "einem", "eines"]
def_articles = ["der", "die", "das"]
```

```python
from collections import Counter
from more_itertools import divide
data = {}
for idx, row in tqdm(corpus.iterrows()):
    lemmas = tokenize_lemmatized(row["lemmatized_text"])
    parts = divide(100, lemmas)
    indef_shares = []
    def_shares = []
    for part in parts:
        lemma_freqs = Counter(part)
        indef_share = sum(lemma_freqs.get(art, 0) for art in indef_articles) / sum(lemma_freqs.values())
        def_share = sum(lemma_freqs.get(art, 0) for art in def_articles) / sum(lemma_freqs.values())
        indef_shares.append(indef_share)
        def_shares.append(def_share)
    data[idx] = {"indef_shares": indef_shares, "def_shares": def_shares}
total_indef = np.array([np.array(entry["indef_shares"]) for entry in data.values()]).mean(axis=0)
total_def = np.array([np.array(entry["def_shares"]) for entry in data.values()]).mean(axis=0)
```

```python
# (Ro)Bert(a)
```

```python
# Sample paragraphs
data = []
for idx, row in tqdm(corpus.iterrows()):
    title = row["title"]
    author = row["author"]
    epoch = row["epoch"]
    text = row["text"]
    
    pargraphs = re.split(r"(\n[ ]){1,}?", text)
    for paragraph in pargraphs:
        if not paragraph.strip().split():
            continue
        par_n_sents = len(sent_tokenize(paragraph, language="german"))
        if par_n_sents < 3:
            continue
        if "»" in paragraph or '"' in paragraph or "'" in paragraph:
            continue
        data.append({
            "title": title,
            "author": author,
            "epoch": epoch,
            "paragraph": paragraph,
            "n_sents": par_n_sents
        })
df = pd.DataFrame.from_records(data)
```

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("uklfr/gottbert-base")

df["n_subword_tokens"] = df.paragraph.progress_apply(lambda text: len(tokenizer(text)["input_ids"]))
df = df.query("n_subword_tokens <= 490")
```

```python
df.shape
```

```python
df.epoch.value_counts()
```

```python
train_data.head()
```

```python
sns.histplot(hue="epoch", x="n_sents", data=df)
```

```python
for idx, row in df.sample(20).iterrows():
    print(f"{row['author']} - {row['title']} - {row['epoch']}")
    print("_"*30)
    print()
    print(row["paragraph"])
    print("#"*60)
```

```python
train_data = pd.concat([
    df.query("epoch == 'romantik'").sample(2000, random_state=42),
    df.query("epoch == 'realismus'").sample(2000, random_state=42)
])
train_data.to_csv("../data/pargraphs_train_wo_ds.csv", index=False)
```

```python
sns.histplot(hue="epoch", x="n_sents", data=train_data)
```

```python

```
