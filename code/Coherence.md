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
from tqdm.auto import tqdm
from random import shuffle, seed, choices
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize
from more_itertools import flatten
from pathlib import Path
import re
tqdm.pandas()
```

```python
STOPWORDS = set(Path("../data/stopwords-de.txt").read_text().split("\n"))
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
df["sents"] = df.lemmatized_text.progress_apply(lambda text: sent_tokenize(text, language="german"))
```

```python
df["paragraphs"] = [[p.strip() for p in part if p.strip()] for part in df.text.str.split(r"(\n[ ]){2,}?")]
#df["paragraphs"] = [[p.strip() for p in part if p.strip()] for part in df.text.str.split(r"\n+")]
```

```python
all_overlaps = []
for index, row in df.iterrows():
    paragraphs = row["sents"]
    type_overlaps = [] 
    for par_idx, paragraph in enumerate(paragraphs):
        curr_paragraph_types = set(token for token in re.findall(r"\w+", paragraph.lower()) if token not in STOPWORDS)
        if par_idx == 0:
            prev_paragraph_types = curr_paragraph_types.copy()
            type_overlaps.append(0.0)
            continue
        if len(curr_paragraph_types) == 0:
            continue
        overlap = len(curr_paragraph_types.intersection(prev_paragraph_types)) / (len(prev_paragraph_types) + 0.000000001)
        type_overlaps.append(overlap)
        prev_paragraph_types = curr_paragraph_types.copy()
    df.loc[index, "mean_overlap"] = np.mean(type_overlaps)
    all_overlaps.append(type_overlaps)

```

```python
df["mean_overlap"].describe()
```

```python
sns.histplot(x="mean_overlap", hue="epoch", data=df)
```

```python
sns.boxenplot(x="epoch", y="mean_overlap", data=df)
```

```python
max_length = max(map(len, all_overlaps))
overlap_df = pd.DataFrame(np.array([i + [0]*(max_length-len(i)) for i in all_overlaps]))
overlap_df[["epoch", "author", "title"]] = df[["epoch", "author", "title"]]
```

```python
plot_df = overlap_df.melt(id_vars=["epoch", "author", "title"], var_name="par_idx")
plot_df
```

```python
sns.lineplot(x="par_idx", y="value", hue="epoch", data=plot_df.query("par_idx <= 100"))
```

```python
idx = overlap_df.query("title == 'Ruhe ist die erste Bürgerpflicht'").index.item()
o = all_overlaps[idx]
plt.scatter(list(range(len(o))), o)
```

```python
overlap_df.query("title == 'Ruhe ist die erste Bürgerpflicht'").index.item()
```

```python
for idx, overlaps in enumerate(all_overlaps):
    overlaps = overlaps[:20]
    epoch = df.iloc[idx].epoch
    plt.plot(list(range(len(overlaps))), overlaps, color="red" if epoch == "romantik" else "green")
plt.show()
```

```python

```

```python
df["rel_nacht"] = df.lemmatized_text.str.lower().str.count("nacht") / df.lemmatized_text.str.findall(r"\w+").apply(lambda x: len(x))
df["rel_tag"] = df.lemmatized_text.str.lower().str.count("tag") / df.lemmatized_text.str.findall(r"\w+").apply(lambda x: len(x))
```

```python
sns.boxplot(y="rel_musik", x="epoch", data=df)
```

```python
begriff_df = df.melt(id_vars=["epoch", "pub_year_estim"], value_vars=["rel_nacht", "rel_tag"])
begriff_df
```

```python
sns.scatterplot(x="pub_year_estim", y="value", hue="variable", data=begriff_df)
```

```python

```
