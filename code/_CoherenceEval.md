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
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments
from datasets import load_from_disk
from model import SentenceOrderingTrainer, SoDataCollator, make_compute_metrics_func, make_tokenization_func, SlidingTokenShuffle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import pandas as pd
```

```python
model = AutoModelForTokenClassification.from_pretrained("so/final_models/rom_rea/bert-base-german-cased/second")
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
```

```python
test_dataset = load_from_disk("../data/rom_rea_hf_dataset")["test"]
```

```python
map_func = SlidingTokenShuffle(
    tokenizer=tokenizer,
    text_column="text",
    target_column="labels",
    window=512,
    padding=160,
    stride=128,
    norm_targets=True
)
test_dataset = test_dataset.map(
    map_func,
    batched=True,
    batch_size=10,
    num_proc=12,
    load_from_cache_file=False,
)
test_dataset.set_format("torch")
```

```python
metrics_func = make_compute_metrics_func(tokenizer.cls_token_id)
trainer = SentenceOrderingTrainer(
    model=model,
    args=TrainingArguments(output_dir="eval_output", per_device_eval_batch_size=64),
    target_token_id=tokenizer.cls_token_id,
    data_collator=SoDataCollator(tokenizer=tokenizer),
    #compute_metrics=metrics_func,
)
```

```python
predictions = trainer.predict(test_dataset)
```

```python
labels = predictions[1]
labels = labels.copy()
if map_func.norm_targets:
    int_labels = []
    for entry in labels:
        entry = entry.copy()
        entry[entry != -100] = entry[entry != -100].argsort()
        int_labels.append(entry)
labels = np.array(int_labels, dtype="int64")
labels
```

```python
input_ids = test_dataset["input_ids"]
padded_input_ids = []
for entry in input_ids:
    n_pads = 512 - len(entry)
    padded_input_ids.append(entry.tolist() + [-100] * n_pads)

input_ids = np.array(padded_input_ids)
input_ids.shape
```

```python
logits = predictions[0].squeeze(2)
```

```python
logits.shape
```

```python
losses = []
preds = []
for i in range(logits.shape[0]):
    lts = logits[i][input_ids[i] == tokenizer.cls_token_id].ravel()
    true = labels[i][labels[i] != -100][:len(lts)] # if our input got truncated due to seq length of model
    mse = ((true - lts)**2).mean()
    pred = np.argsort(np.argsort(lts))
    preds.append(pred)
    losses.append(mse)
losses = np.array(losses)
losses.mean()
```

```python
sns.boxplot(y=losses)
```

```python
from scipy.stats import kendalltau, wilcoxon
from sklearn.metrics import accuracy_score

acc_scores = []
taus = []
for i in tqdm(range(len(preds))):
    p = preds[i]
    t = labels[i][labels[i] != -100][:len(p)]
    acc = accuracy_score(t, p)
    acc_scores.append(acc)
    tau, pvalue = kendalltau(t, p)
    taus.append(tau)

acc_scores = np.array(acc_scores)
taus = np.array(taus)
```

```python
taus = np.array(taus)
```

```python
taus.shape
```

```python
acc_scores.mean()
```

```python
taus[~np.isnan(taus)].mean()
```

```python
test_df = test_dataset.to_pandas()
test_df.head(2)
```

```python
test_df["loss"] = losses
test_df["acc"] = acc_scores
test_df["tau"] = taus
```

```python
len(test_df.query("n_shuffled_sents > 5 and acc == 1.0")) / len(test_df)
```

```python
test_df.query("acc == 1.0").groupby("epoch")["title"].count() / test_df.groupby("epoch")["title"].count()
```

```python
sns.boxplot(x="epoch", y="acc", data=test_df)
```

```python
sns.boxplot(x="epoch", y="tau", data=test_df)
```

```python
sns.violinplot(x="epoch", y="acc", data=test_df)
```

```python
sns.violinplot(x="epoch", y="tau", data=test_df)
```

```python
test_df.groupby("epoch")["acc"].describe()
```

```python
test_df.groupby("epoch")["tau"].describe()
```

```python
test_df.query("n_shuffled_sents == 5").groupby("epoch")["acc"].describe()
```

```python
test_df.query("n_shuffled_sents == 5").groupby("epoch")["tau"].describe()
```

```python
from math import factorial
sns.catplot(x="n_shuffled_sents", y="acc", hue="epoch", kind="point", data=test_df, height=8.27, aspect=11.7/8.27)
#sns.lineplot(x=list(range(1,64)), y=[1 / factorial(i) for i in range (1,64)])
```

```python
sns.catplot(x="n_shuffled_sents", y="tau", hue="epoch", kind="point", data=test_df, height=8.27, aspect=11.7/8.27)
```

```python
test_df.groupby("epoch").n_shuffled_sents.describe()
```

```python
sns.boxplot(y="n_shuffled_sents", x="epoch", data=test_df)
```

```python
from scipy.stats import wilcoxon, ttest_ind
for _ in range(10):
    wilcoxon_result = wilcoxon(
        test_df.query("epoch == 'romantik'")["acc"],
        test_df.query("epoch == 'realismus'").sample(len( test_df.query("epoch == 'romantik'")))["acc"]
    )
    ttest_result = ttest_ind(
        test_df.query("epoch == 'romantik'")["acc"],
        test_df.query("epoch == 'realismus'").sample(len( test_df.query("epoch == 'romantik'")))["acc"]
    )
    print(wilcoxon_result)
    print(ttest_result)
    print()
```

```python
from scipy.stats import wilcoxon, ttest_ind
for _ in range(10):
    wilcoxon_result = wilcoxon(
        test_df[~np.isnan(test_df.tau)].query("epoch == 'romantik'")["tau"],
        test_df[~np.isnan(test_df.tau)].query("epoch == 'realismus'").sample(len( test_df[~np.isnan(test_df.tau)].query("epoch == 'romantik'")))["acc"]
    )
    ttest_result = ttest_ind(
        test_df[~np.isnan(test_df.tau)].query("epoch == 'romantik'")["tau"],
        test_df[~np.isnan(test_df.tau)].query("epoch == 'realismus'").sample(len( test_df[~np.isnan(test_df.tau)].query("epoch == 'romantik'")))["acc"]
    )
    print(wilcoxon_result)
    print(ttest_result)
    print()
```

```python
sns.histplot(x="acc", hue="epoch", data=test_df)
```

```python
sns.histplot(x="tau", hue="epoch", data=test_df)
```

```python
sns.scatterplot(x="acc", y="tau", hue="epoch", data=test_df)
```

```python
sns.kdeplot(x="n_shuffled_sents", hue="epoch", data=test_df)
```

```python
test_df.query("epoch == 'romantik'").groupby("title")["acc"].mean().sort_values()
```

```python
test_df.query("epoch == 'romantik'").groupby("title")["title"].count().sort_values()
```

```python
for idx, row in test_df.query("epoch == 'romantik'").sort_values(by="acc").iloc[::2500].iterrows():
    print(f"{row['author']}-{row['title']}: Acc:{row['acc']}")
    print("_"*60)
    print(row["text"])
    print("#"*60)
    print()
```

```python
for idx, row in test_df.query("epoch == 'realismus'").sort_values(by="acc").iloc[::10000].iterrows():
    print(f"{row['author']}-{row['title']}: {row['acc']}")
    print("_"*60)
    print(row["text"])
    print("#"*60)
    print()
```

```python
test_df["n_shuffled_sents"].describe()
```

```python tags=[]
from pathlib import Path

for title in test_df.title.unique():
    title_df = test_df.query("title == @title").sort_values(by="par_idx")
    rolling_df = title_df.rolling(10).mean()
    g = sns.relplot(x="par_idx", y="tau", kind="line", data=rolling_df)
    name = f"{title_df.author.iloc[0]} | {title_df.title.iloc[0]} | {title_df.epoch.iloc[0]}"
    g.figure.suptitle(name, y=1.05)
    file_path = Path("../plots/by_title/") / "_".join(name.split())
    g.savefig(str(file_path.resolve()))
```

```python
test_df.query("epoch == 'romantik'").groupby("par_idx").tau.std()
```

```python
sns.lineplot(
    x="window_index", y="acc",
    data=test_df.query("epoch == 'romantik'").groupby("window_index").acc.mean().to_frame().rolling(100).mean()
)
```

```python
sns.lineplot(
    x="window_index", y="acc",
    data=test_df.query("epoch == 'realismus'").groupby("window_index").acc.mean().to_frame().rolling(100).mean()
)
```

```python
test_df.pivot(index="par_idx", columns=["author", "title", "epoch"], values=["acc", "tau"])
```

```python
title_df.sort_values(by="par_idx", ascending=False)
```

```python
title_df.par_idx
```

```python
test_df.query("title == 'Die Günderode'").sort_values(by="par_idx")
```

```python
rea_mean = test_df.query("epoch == 'realismus'").groupby("window_index").acc.mean().to_frame().reset_index()
rea_mean["epoch"] = "realismus"
rea_mean
```

```python
rom_mean = test_df.query("epoch == 'romantik'").groupby("window_index").acc.mean().to_frame().reset_index()
rom_mean["epoch"] = "romantik"
rom_mean
```

```python
comb = pd.concat([rom_mean, rea_mean]).reset_index()
comb
```

```python
sns.scatterplot(x="window_index", y="acc", hue="epoch", data=comb)
```

```python
data = []
for title in test_df.title.unique():
    title_df = test_df.query("title == @title").query("n_shuffled_sents == 5")
    if len(title_df) == 0:
        continue
    chunks = np.array_split(title_df, 100)
    entry = {"title": title, "epoch": title_df.epoch.iloc[0]}
    for i, c in enumerate(chunks):
        entry[i] = c[~np.isnan(c.tau)].tau.mean()
    data.append(entry)
chunk_df = pd.DataFrame.from_records(data)
chunk_df_longform = chunk_df.melt(id_vars=["title", "epoch"], var_name="segment", value_name="mean_tau")
```

```python
chunk_df_longform.head(3)
```

```python
sns.lineplot(x="segment", y="mean_tau", hue="epoch", data=chunk_df_longform)
```

```python

```

```python

```
