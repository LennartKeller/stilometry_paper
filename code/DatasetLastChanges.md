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

1. Assign some authors to the other epoch.
2. Remove Stendhal
3. Remove all texts with pub_year_estim == 0 && pub_year_estim > 1895

```python
import pandas as pd
from nltk import sent_tokenize
from tqdm.auto import tqdm
tqdm.pandas()
```

```python
df = pd.read_csv("../data/rom_real_dataset.csv")
```

```python
df.loc[df.author == 'Gotthelf,-Jeremias', 'epoch'] = "realismus"
df.loc[df.author == 'Droste-Huelshoff,-Annette-von', 'epoch'] = "realismus"
```

```python
df = df.drop(df.query("author == 'Stendhal'").index)
df = df.drop(df.query("author == 'Balzac,-Honore-de'").index)
df = df.drop(df.query("author == 'Scheerbart,-Paul'").index)
df = df.drop(df.query("author == 'Thoma,-Ludwig'").index)
df = df.drop(df.query("author == 'Meyrink,-Gustav'").index)
df = df.drop(df.query("author == 'Meyrink,-Gustav'").index)
df = df.drop(df.query("author == 'Spiegel,-Karl'").index)
df = df.drop(df.query("author == 'Grimm,-Albert-Ludewig'").index)

df = df.drop(df.query('title == "1. Mordi\'s Garten"').index)
df = df.drop(df.query('title == "1. Schneewittchen"').index)
```

```python
df["n_sents"] = df.normed_text.progress_apply(
    lambda text: len(sent_tokenize(text, language="german"))
)
df = df.query("n_sents > 10")
```

```python
df = df.query("pub_year_estim > 0")
```

```python
df.reset_index().to_csv("../data/rom_real_dataset_final.csv", index=False)
```

```python

```
