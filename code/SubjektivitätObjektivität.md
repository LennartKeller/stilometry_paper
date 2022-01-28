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
import matplotlib.pyplot as plt
import seaborn as sns
#from lib import *
from tqdm.auto import tqdm
from random import shuffle, seed, choices
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize
from more_itertools import flatten
```

```python jupyter={"source_hidden": true} tags=[]
tqdm.pandas()
sns.set_theme()
```

```python
RANDOM_SEED = 42
seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

```python
df = pd.read_csv("../data/rom_real_dataset_final.csv")
df.shape
```

```python
df.head()
```

# Unterschiede

* Texte der Romantik sind gekennzeichnet durch Elemente des Idyllischen und/oder Phantastischen und/oder eine idealisierende Darstellungsweise und richten ihren Fokus auf das subjektive emotionale Empfinden und Erleben einzelner Figuren, sowie die damit assoziierten psychischen und seelischen Prozesse.

* Texte des Realismus sind gekennzeichnet durch ihre Wirklichkeitszugewandtheit und das Ziel, jene Wirklichkeit detailgetreu, anschaulich und tendenziell objektiv darzustellen. Sie fassen das Individuum als stets beeinflusst durch gesellschaftliche Normen, überpersönliche Ereignisse und soziales Milieu auf. Außerdem richten sie ihren Fokus auf die Darstellung dieses Sachverhalts.

```python
#from textblob_de import TextBlobDE as TextBlob
#all_sentiments = []
#for _, row in tqdm(list(df.iterrows())):
#    text_sentiments = []
#    text = row["text"]
#    for sent in sent_tokenize(text, language="german"):
#        blob = TextBlob(sent)
#        sentiment = blob.sentiment
#        text_sentiments.append(sentiment)
#    all_sentiments.append({
#        "index": row["index"],
#        "sentence_sentiments": text_sentiments
#    })
```

```python
#import json
#
#json_sentiments = [
#    {
#        "index": entry["index"],
#        "polarity": [i.polarity for i in entry["sentence_sentiments"]],
#        "subjectivity": [i.subjectivity for i in entry["sentence_sentiments"]]
#    }
#    for entry in all_sentiments
#]
#
#with open("sentence_sentiments.json", "w") as f:
#    json.dump(json_sentiments, f, indent=4)
```

```python
merge_df = pd.read_json("sentence_sentiments.json")
merge_df["polarity_zero_share"] = merge_df["polarity"].apply(lambda values: len(np.array(values)[np.array(values) == 0.0]) / len(values))
merge_df["polarity_mean"] = merge_df["polarity"].apply(lambda values: np.mean(values))
merge_df["subjectivity_zero_share"] = merge_df["subjectivity"].apply(lambda values: len(np.array(values)[np.array(values) == 0.0]) / len(values))
merge_df["subjectivity_mean"] = merge_df["subjectivity"].apply(lambda values: np.mean(values))
merge_df["polarity_std"] = merge_df["polarity"].apply(lambda values: np.std(values))
merge_df["subjectivity_std"] = merge_df["subjectivity"].apply(lambda values: np.std(values))
```

```python
df = df.merge(merge_df, on="index")
```

```python
sns.boxenplot(x="epoch", y="polarity_mean", data=df)
```

```python
sns.boxenplot(x="epoch", y="polarity_std", data=df)
```

```python
sns.boxenplot(x="epoch", y="polarity_zero_share", data=df)
```

```python
sns.boxenplot(x="epoch", y="subjectivity_mean", data=df)
```

```python
sns.boxenplot(x="epoch", y="subjectivity_std", data=df)
```

```python
sns.boxenplot(x="epoch", y="subjectivity_zero_share", data=df)
```

```python
sns.lmplot(x="polarity_mean", y="polarity_std", hue="epoch", data=df)
```

```python
sns.lmplot(x="polarity_mean", y="polarity_zero_share", hue="epoch", data=df)
```

```python
sns.lmplot(x="subjectivity_mean", y="subjectivity_std", hue="epoch", data=df)
```

```python
sns.lmplot(x="subjectivity_mean", y="subjectivity_zero_share", hue="epoch", data=df)
```

```python
sns.lmplot(x="pub_year_estim", y="polarity_mean", hue="epoch", data=df)
```

```python
sentiment_df = pd.read_json("sentence_sentiments.json")
```

```python
sentiment_df["epoch"] = df["epoch"]
```

```python
sentiment_df.query("epoch == 'realismus'")
```

```python
import re
all_rom_pars = []
all_rea_parts = []
for index, row in df.iterrows():
    tagged_text = row["tagged_text"]
    tags = [
        {"token": match.group(1), "pos_tag": match.group(2), "ner_tag": ""} if not "/" in match.group(2) else
        {"token": match.group(1), "pos_tag": match.group(2).split("/")[0], "ner_tag": match.group(2).split("/")[1]}
        for match in re.finditer(r"([^\s]+?) <(.+?)>", tagged_text)
    ]
    
    df["only_nouns"] = []
    
    
```

```python

```

```python

```
