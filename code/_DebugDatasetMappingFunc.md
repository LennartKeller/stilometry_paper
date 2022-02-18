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
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm.auto import tqdm
tqdm.pandas()
import pandas as pd
```

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
```

```python
from model import SlidingTokenShuffle
map_func = SlidingTokenShuffle(tokenizer=tokenizer,
                               text_column="text",
                               target_column="labels",
                               window=512,
                               padding=160,
                               stride=128
                              )
```

```python
dataset = load_from_disk("../data/rom_rea_hf_dataset")["train"]
dataset
```

```python
shuffled_dataset = dataset.map(
    map_func,
    batched=True,
    batch_size=10,
    num_proc=12,
    load_from_cache_file=False
)
```

```python
shuffled_dataset
```

```python
df = shuffled_dataset.to_pandas()
df.n_shuffled_sents.describe()
```

```python
df.groupby("title").n_shuffled_sents.describe()
```

```python
for text in df.text.sample(5):
    print(text)
    print()
```

```python
"... [CLS] Das ist ein Satz. [CLS] Dies auch! [CLS] Der bricht a".split("[CLS]")
```

```python
import numpy as np
x = np.ones((16,512))
big_x = np.repeat(np.expand_dims(x, 0), 128, 0)
big_x.shape
```

```python
import numpy as np
x = np.ones((16,512))
big_x = np.tile(x, (128, 0))
big_x.shape
```

```python
import numpy as np
x = np.ones((16,512))
big_x_only_view = x.view((128, 16, 512))
```

```python

```
