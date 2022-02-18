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
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer, TrainingArguments
from datasets import load_from_disk

from model import SentenceOrderingTrainer, SoDataCollator, make_compute_metrics_func, make_tokenization_func, SlidingTokenShuffle

from tqdm.auto import tqdm
tqdm.pandas()
```

```python
test_dataset = load_from_disk("../data/rom_rea_hf_dataset")["test"]
```

```python
model = AutoModel.from_pretrained(
    "so/final_models/rom_rea/bert-base-german-cased/second"
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
```

```python
test_df = test_dataset.to_pandas()
```

```python
test_df
```

```python
from model import SentEmbeddingExtractor
```

```python
sent_embeddings_extractor = SentEmbeddingExtractor(model=model, tokenizer=tokenizer, batch_size=16, device="cuda:0")
```

```python
test_df.sample(1)
```

```python
text = test_df.iloc[20].text
sentence_embeddings = sent_embeddings_extractor.extract(text)
```

```python
sents = sent_embeddings_extractor.sent_tokenize(text)

dists = []
length_quot = []
for i in range(sentence_embeddings.shape[0]):
    e = sentence_embeddings[i]
    if i == 0:
        dists.append(0.0)
        length_quot.append(1.0)
    else:
        dists.append(np.linalg.norm(e-sentence_embeddings[i-1]))
        length_quot.append(len(sents[i].split())/len(sents[i-1].split()))
dists = np.array(dists)
sent_df = pd.DataFrame(dict(sent=sents, position=list(range(len(dists))), dist=dists, length_quot=length_quot))
sent_df["length_quot_log"] = np.log(length_quot)
```

```python
sns.lineplot(x="position", y="dist", data=sent_df.rolling(1).mean())
```

```python
sns.displot(x="length_quot_log", y="dist", kind="kde",data=sent_df)
```

```python
for index, row in sent_df.iloc[:50].iterrows():
    #print("↑"+"_"*80+"↑")
    print()
    print("↑--"+str(row["dist"])+"--↓")
    print()
    #print("↓"+"_"*80+"↓")
    print(row["sent"])
```

```python
sent_df
```

```python
sns.histplot(x="length_quot_log", data=sent_df)
```

```python

```
