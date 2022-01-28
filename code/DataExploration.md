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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import umap.plot as umap_plot
from sklearn.feature_extraction.text import TfidfVectorizer
from lib import get_stopwords, tokenize_lemmatized
from zeta import Zeta
from string import punctuation
```

```python
STOPWORDS = get_stopwords()
PUNCTUATION = set(list(punctuation) + ["..."]) 
```

```python
df = pd.read_csv("../data/rom_real_dataset_final.csv")
df.shape
```

### UMAP Visualisierung Tfidf

```python
tfidf = TfidfVectorizer(
    tokenizer=tokenize_lemmatized,
    stop_words=STOPWORDS,
    max_features=20_000,
    ngram_range=(1,3)
)
X_tfidf = tfidf.fit_transform(df.lemmatized_text)
```

```python
umap = UMAP(n_jobs=-1, n_components=2, min_dist=0.001, n_neighbors=5)

Xr = umap.fit_transform(X_tfidf)
```

```python
umap_plot.points(umap, labels=df.epoch, theme="fire")
```

```python
umap_plot.output_notebook()
p = umap_plot.interactive(umap, labels=df.epoch, hover_data=df[["title", "author"]], point_size=5)
umap_plot.show(p)
```

## Zeta: Distinktive Wörter für beide Epochen

```python
X_zeta = df.lemmatized_text.apply(
    lambda text: [
        token
        for token in tokenize_lemmatized(text)
        if token not in STOPWORDS and token not in PUNCTUATION and not token.isdigit()
    ]
).to_list()
y_zeta = df.epoch.to_list()
```

```python
zeta = Zeta()
```

```python
zeta_scores = zeta.fit(X_zeta, y_zeta, target_partition="romantik")
```

```python
zeta.vocab_[np.argsort(zeta_scores)]
```

```python
zeta_df = pd.DataFrame({
    "token": zeta.vocab_,
    "score": zeta_scores
})
```

```python
zeta_top_10 = pd.concat([
    zeta_df.sort_values(by="score", ascending=False).head(15),
    zeta_df.sort_values(by="score", ascending=False).tail(15)
])
```

```python
zeta_top_10
```

```python

```
