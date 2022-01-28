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
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

tfidf = TfidfVectorizer(max_features=10_000)

X = tfidf.fit_transform(df.lemmatized_text)
Xr = UMAP(n_components=100).fit_transform(X.todense())
```

```python
from sklearn.cluster import AffinityPropagation

ap = AffinityPropagation(damping=0.9)

clusters = ap.fit_predict(X)
ap.cluster_centers_indices_.shape

```

```python
df.iloc[ap.cluster_centers_indices_].epoch.value_counts()
```

```python
df["cluster_label"] = clusters
```

```python
for cluster_idx in df.cluster_label.unique():
    print(cluster_idx)
    cluster_df = df.query("cluster_label == @cluster_idx")
    print(cluster_df.epoch.value_counts())
    print(cluster_df[["author", "title", "epoch"]])
    print()
```

```python
df.head()
```

```python

```
