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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine, euclidean, minkowski
from umap import UMAP
```

```python
test_df = load_from_disk("../data/rom_rea_hf_dataset")["test"].to_pandas()
```

```python
model = AutoModel.from_pretrained(
    "so/final_models/rom_rea/bert-base-german-cased/second"
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
```

```python
from typing import List
import torch
import numpy as np
from nltk import sent_tokenize
from transformers import PreTrainedModel, PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding
from tqdm.auto import tqdm

class SentenceEncoder:
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 16,
        language: str = "german",
        device: str = "cpu",
        verbose: bool = False
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.language = language
        self.device = device
        self.verbose = verbose
    
    def sent_tokenize(self, text: str) -> List[str]:
        return sent_tokenize(text, language=self.language)
    
    def prepare_sent(self, sent: str) -> str:
        return f"{self.tokenizer.cls_token} {sent}"
    
    def encode_sent(self, prepared_sent: List[str]) -> List[BatchEncoding]:
        inputs = self.tokenizer(
            prepared_sent,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True
        )
        inputs = BatchEncoding({
            key: value.reshape(-1)
            for key, value in inputs.items()
        })
        return inputs
        
    
    def encode(self, text: str) -> np.ndarray:
        sents = self.sent_tokenize(text)
        prepared_sents = [self.prepare_sent(sent) for sent in sents]
        encoded_sents = [self.encode_sent(sent) for sent in prepared_sents]
        dataloader = torch.utils.data.DataLoader(
            encoded_sents,
            batch_size=self.batch_size,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
        )
        all_sentence_embeddings = []
        if self.verbose:
            pbar = tqdm(dataloader, desc="Encoding sents")
        else:
            pbar = dataloader
        for batch in pbar:
            batch = batch.to(self.device)
            with torch.no_grad():
                outputs = self.model(**batch)
            last_hidden_state = outputs["last_hidden_state"]
            input_ids = batch["input_ids"]
            sentence_embeddings = []
            for i, l in zip(input_ids, last_hidden_state):
                sent_embedding = l[i == self.tokenizer.cls_token_id].detach().cpu().numpy()
                sentence_embeddings.append(sent_embedding)
            all_sentence_embeddings.extend(sentence_embeddings)
        return np.array(all_sentence_embeddings)
```

```python
sent_encoder = SentenceEncoder(
    model=model,
    tokenizer=tokenizer,
    batch_size=128,
    device="cuda:0"
)
```

```python
data = {}
for index, row in tqdm(test_df.iterrows(), desc="Encoding dataset"):
    text = row["text"]
    sents = sent_encoder.sent_tokenize(text)
    sentence_embeddings = sent_encoder.encode(text)
    
    # Compute df for each title
    cosine_dists = []
    euclidean_dists = []
    manhattan_dists = []
    
    length_quot = []
    for i in range(sentence_embeddings.shape[0]):
        e = sentence_embeddings[i]
        if i == 0:
            cosine_dists.append(0.0)
            euclidean_dists.append(0.0)
            manhattan_dists.append(0.0)
            length_quot.append(1.0)
        else:
            cosine_dists.append(cosine(e, sentence_embeddings[i-1]))
            euclidean_dists.append(euclidean(e, sentence_embeddings[i-1]))
            manhattan_dists.append(minkowski(e, sentence_embeddings[i-1], p=1))
            length_quot.append(len(sents[i].split())/len(sents[i-1].split()))
    
    cosine_dists = np.array(cosine_dists)
    euclidean_dists = np.array(euclidean_dists)
    manhattan_dists = np.array(manhattan_dists)
    
    sent_df = pd.DataFrame(dict(
        sent=sents,
        position=list(range(len(cosine_dists))),
        cosine_dist=cosine_dists,
        euclidean_dist=euclidean_dists,
        manhattan_dist=manhattan_dists,
        length_quot=length_quot
    ))
    
    sent_df["length_quot_log"] = np.log(length_quot)
    sent_df["perc"] = sent_df["position"] / sent_df["position"].max()
    
    data[index] = {"sent_df": sent_df, "sentence_embeddings": sentence_embeddings} 
```

```python
test_df
```

```python
print(*sorted(data.keys()), sep=" | ")
```

```python
INDEX = 150
sent_df = data[INDEX]["sent_df"]
test_df.iloc[INDEX].title, test_df.iloc[INDEX].author, test_df.iloc[INDEX].epoch
```

```python
sns.lineplot(x="position", y="euclidean_dist", data=sent_df.rolling(len(sent_df) // 10).mean())
```

```python
for index, row in sent_df.iloc[:50].iterrows():
    #print("↑"+"_"*80+"↑")
    print()
    print("↑--"+str(row["euclidean_dist"])+"--↓")
    print()
    #print("↓"+"_"*80+"↓")
    print(row["sent"])
```

```python
df_data = []
for index, entries in data.items():
    epoch = test_df.iloc[index]["epoch"]
    sent_df = entries["sent_df"].copy()
    sent_df["epoch"] = [epoch] * len(sent_df)
    df_data.append(sent_df)
all_df = pd.concat(df_data)
all_df
```

```python
sns.scatterplot(x="perc", y="euclidean_dist", hue="epoch", data=all_df.sample(frac=0.1))
```

```python
all_df["p"] = all_df["perc"].apply(lambda perc: round(perc * 100))
all_df["tenths"] = all_df["perc"].apply(lambda perc: round(perc * 100)) // 20
```

```python tags=[]
#sns.lineplot(x="tenths", y="euclidean_dist", hue="epoch", data=all_df)
```

```python
rom_tenth_mean = all_df.query("epoch == 'romantik'").groupby("tenths")["euclidean_dist"].mean() / all_df.query("epoch == 'romantik'").groupby("tenths")["euclidean_dist"].mean().sum()
rom_tenth_std = all_df.query("epoch == 'romantik'").groupby("tenths")["euclidean_dist"].std()
rea_tenth_mean = all_df.query("epoch == 'realismus'").groupby("tenths")["euclidean_dist"].mean() / all_df.query("epoch == 'realismus'").groupby("tenths")["euclidean_dist"].mean().sum()
rea_tenth_std= all_df.query("epoch == 'realismus'").groupby("tenths")["euclidean_dist"].std()
```

```python
plt.plot(rom_tenth_mean)
plt.plot(rea_tenth_mean)
plt.show()
```

```python

```
