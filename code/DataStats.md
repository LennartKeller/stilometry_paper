---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lib import *
sns.set_theme()
```

```{code-cell} ipython3
df = pd.read_csv("../data/rom_real_dataset_final.csv")
df.shape
```

```{code-cell} ipython3
df.head(10)
```

```{code-cell} ipython3
df["n_tokens"] = df.lemmatized_text.apply(
    lambda text: len(tokenize_lemmatized(text))
)
```

```{code-cell} ipython3
df_rom = df.query("epoch == 'romantik'")
df_rea = df.query("epoch == 'realismus'")
df_rom.shape, df_rea.shape
```

```{code-cell} ipython3
df_rom.groupby("author").title.count().sort_values().plot.bar()
```

```{code-cell} ipython3
author_plot_df_rom = df_rom.groupby("author")["title"].count().to_frame().merge(
    df_rom.groupby("author")["n_tokens"].sum().to_frame(), on="author"
).sort_values("n_tokens")
```

```{code-cell} ipython3
rom_title_token_plot = author_plot_df_rom.plot(kind='bar', secondary_y='n_tokens', rot=90)
rom_title_token_plot.set_title("Romantik")
rom_title_token_plot.figure.savefig("../plots/rom_title_token_plot.png", dpi=300, bbox_inches='tight')
```

```{code-cell} ipython3
author_plot_df_rea = df_rea.groupby("author")["title"].count().to_frame().merge(
    df_rea.groupby("author")["n_tokens"].sum().to_frame(), on="author"
).sort_values("n_tokens")
```

```{code-cell} ipython3
rea_title_token_plot = author_plot_df_rea.plot(kind='bar', secondary_y='n_tokens', rot=90)
rea_title_token_plot.set_title("Realismus")
rea_title_token_plot.figure.savefig("../plots/rea_title_token_plot.png", dpi=300, bbox_inches='tight')
```

### Vergleiche Anzahl der Tokens pro Epoche

```{code-cell} ipython3
df.groupby("epoch").n_tokens.sum()
```

```{code-cell} ipython3
# Make latex table with number of author pro epooch and #tokens
sns.barplot(x="epoch", y="n_tokens", data=df)
```

```{code-cell} ipython3
text_length_plot = sns.violinplot(x="epoch", y="n_tokens", data=df, size=2.3)
text_length_plot.set_title("Verteilung der Textlängen pro Epoche.")
text_length_plot.figure.savefig("../plots/text_length_plot.png", dpi=300, bbox_inches='tight')
```

### Werke pro Jahr

```{code-cell} ipython3
sns.stripplot(x="pub_year_estim", y="epoch", data=df)
```

```{code-cell} ipython3
df.sort_values("pub_year_estim").query("epoch == 'romantik'")
```

```{code-cell} ipython3
df.sort_values("pub_year_estim").query("epoch == 'realismus'")
```

```{code-cell} ipython3

```
