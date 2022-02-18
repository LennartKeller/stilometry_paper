---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
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
from string import punctuation
tqdm.pandas()
```

```python
PUNCTUATION = set(punctuation + "»«")
"".join(PUNCTUATION)
```

```python
def clean_punctuation(string):
    cleaned_string = str(string)
    for char in PUNCTUATION:
        cleaned_string.replace(char, "")
    return cleaned_string
```

```python
corpus = pd.read_csv("../data/rom_real_dataset_final.csv")
corpus.shape
```

```python
corpus["n_tokens"] = corpus.text.str.findall(r"\w+").apply(len)
```

```python
corpus.groupby("epoch")["title"].count()
```

```python
corpus.groupby("epoch")["n_tokens"].sum()
```

```python
corpus_train = pd.concat([
    corpus.query("epoch == 'romantik'").sample(40, random_state=42),
    corpus.query("epoch == 'realismus'").sample(40, random_state=42)
])
corpus_test = corpus.drop(corpus_train.index)

with open("../data/paragraph_ordering_corpus_train_idx.txt", "w") as f:
    f.write("\n".join(map(str, corpus_train.index)))

with open("../data/paragraph_ordering_corpus_test_idx.txt", "w") as f:
    f.write("\n".join(map(str, corpus_test.index)))
```

```python
print(*map(len, (corpus, corpus_train, corpus_test)))
```

```python
corpus_train.groupby("epoch").n_sents.sum().plot.bar()
```

```python
from collections import Counter
from more_itertools import divide

def create_paragraph_df(corpus: pd.DataFrame) -> pd.DataFrame:
    # Sample paragraphs
    data = []
    for idx, row in tqdm(corpus.iterrows()):
        title = row["title"]
        author = row["author"]
        epoch = row["epoch"]
        text = row["text"]

        #pargraphs = re.split(r"(\n[ ]){2,}?", text)
        pargraphs = re.split(r"\n[ ]*\n", text)
        par_idx = 0
        for paragraph in pargraphs:
            paragraph = paragraph.strip()
            if not paragraph.split():
                continue
            par_n_sents = len(sent_tokenize(paragraph, language="german"))
            if par_n_sents == 1:
                continue
            if not "".join(clean_punctuation(paragraph).split()):
                continue
            data.append({
                "par_idx": par_idx,
                "title": title,
                "author": author,
                "epoch": epoch,
                "paragraph": paragraph,
                "n_sents": par_n_sents
            })
            par_idx += 1
    df = pd.DataFrame.from_records(data)
    return df
```

```python
df_train = create_paragraph_df(corpus_train)
df_test = create_paragraph_df(corpus_test)
```

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

df_train["n_subword_tokens"] = df_train.paragraph.progress_apply(lambda text: len(tokenizer(text)["input_ids"]))
df_train = df_train.query("n_subword_tokens <= 500")

df_test["n_subword_tokens"] = df_test.paragraph.progress_apply(lambda text: len(tokenizer(text)["input_ids"]))
```

```python
print(*map(len, (df_train, df_test)))
```

```python
df_train.epoch.value_counts()
```

```python
sns.histplot(hue="epoch", x="n_sents", data=df_train)
```

```python
df_train.n_subword_tokens.describe()
```

```python
for idx, row in df_test.sample(5).iterrows():
    print(f"{row['author']} - {row['title']} - {row['epoch']}")
    print("_"*30)
    print()
    print(repr(row["paragraph"]))
    print()
    print("#"*60)
    print()
```

```python
df_train.to_csv("../data/paragraphs_train.csv", index=False)
df_test.to_csv("../data/paragraphs_test.csv", index=False)
```

```python
from random import shuffle
from random import seed as set_seed
from nltk import sent_tokenize

def make_shuffle_func(sep_token):
    def shuffle_paragraphs(entries, seed=42):
        set_seed(seed)
        shuffled_paragraphs = []
        target_idx = []
        for paragraph in entries["paragraph"]:
            par_sents = sent_tokenize(paragraph, language="german")
            orig_idx = list(range(len(par_sents)))
            par_sents_with_idx = list(zip(par_sents, orig_idx))
            shuffle(par_sents_with_idx)
            shuffled_paragraph = f"{sep_token} " + f" {sep_token} ".join(
                [s[0] for s in par_sents_with_idx]
            )
            so_targets = [s[1] for s in par_sents_with_idx]
            shuffled_paragraphs.append(shuffled_paragraph)
            target_idx.append(so_targets)
        shuffled_entries = {}
        shuffled_entries["shuffled"] = shuffled_paragraphs
        shuffled_entries["so_targets"] = target_idx
            
        return shuffled_entries

    return shuffle_paragraphs
```

```python
from datasets import Dataset, DatasetDict

train_dataset = Dataset.from_csv("../data/paragraphs_train.csv")
train_dataset = train_dataset.map(make_shuffle_func("<s>"), batched=True)
```

```python
train_test = train_dataset.train_test_split(test_size=0.05, seed=42)

train_dataset = DatasetDict(
    {
        "train": train_test["train"],
        "val": train_test["test"],
    }
)
train_dataset.save_to_disk("../data/rom_rea_so_train_hf")
```

```python
test_dataset = Dataset.from_csv("../data/paragraphs_test.csv")
test_dataset = test_dataset.map(make_shuffle_func("<s>"), batched=True)
test_dataset.save_to_disk("../data/rom_rea_so_test_hf")
```

```python

```
