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
from lib import *
from tqdm.auto import tqdm
from random import shuffle, seed, choices
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize
from more_itertools import flatten
```

```python jupyter={"source_hidden": true} tags=[]
tqdm.pandas()
sns.set_theme()
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
df.head()
```

# Unterschiede

* Texte der Romantik sind gekennzeichnet durch Elemente des Idyllischen und/oder Phantastischen und/oder eine idealisierende Darstellungsweise und richten ihren Fokus auf das subjektive emotionale Empfinden und Erleben einzelner Figuren, sowie die damit assoziierten psychischen und seelischen Prozesse.

* Texte des Realismus sind gekennzeichnet durch ihre Wirklichkeitszugewandtheit und das Ziel, jene Wirklichkeit detailgetreu, anschaulich und tendenziell objektiv darzustellen. Sie fassen das Individuum als stets beeinflusst durch gesellschaftliche Normen, überpersönliche Ereignisse und soziales Milieu auf. Außerdem richten sie ihren Fokus auf die Darstellung dieses Sachverhalts.

<!-- #region tags=[] -->
## Hypothese 1: Satzgestaltung

Wenn das Ziel eine plastische und akkurate Darstellung der Wirklichkeit ist, sollten realistische Texte nicht nur Detailliertheit, sondern auch Verständlichkeit anstreben. 
Dafür spräche eine kürzere Satzlänge und nicht zu komplexe Satzgefüge.

Letzteres ließe sich potenziell anhand der Häufigkeit von Konjunktionen und Subjunktionen erfassen.
<!-- #endregion -->

### Satzlängen

```python
rom_sents = []
rea_sents = []

for index, row in tqdm(list(df.iterrows())):
    sents = sent_tokenize(row["lemmatized_text"], language="german")
    if row["epoch"] == "romantik":
        rom_sents.extend(sents)
    else:
        rea_sents.extend(sents)
    df.loc[index, "n_sents"] = len(sents)
    df.loc[index, "mean_sent_length"] = np.mean([len(re.findall(r"\w+", sent)) for sent in sents])

```

```python
sns.histplot(hue="epoch", x="mean_sent_length", data=df, kde=True)
```

Das Histogram legt nahe, dass Werke des Realismus in Bezug auf die Satzlänge stärker variieren.
Auffällig ist auch, dass es viele Werke gibt, deren Sätze im Schnitt recht kurz sind.

Hier wird allerdings nur die durchschnittliche Satzlänge pro Werk betrachtet. Dieser Wert könnte aber durch mehrere Faktoren beeinflusst werden.

Um zu überprüfen, ob die Satzlänge sich auch ohne konkrete Werke als Bezugrahmen unterscheiden samplen wir jeweils 20.000 zufälllige Sätze aus  beiden Epochen und überprüfen ob, sich anhand dieser Daten auch ein Unterschied festmachen lässt.

```python
K = 20000
seed(21)
sample_sents = choices(rom_sents, k=K) + choices(rea_sents, k=K)
sample_labels = ["romantik"] * K + ["realismus"] * K


sent_length_sample_df = pd.DataFrame({
    "sent_length": [len(re.findall(r"\w+", sent)) for sent in sample_sents],
    "epoch": sample_labels
})
```

```python
sent_length_sample_df.epoch.value_counts()
```

```python
sns.histplot(hue="epoch", x="sent_length", data=sent_length_sample_df, kde=True)
```

```python
sns.violinplot(x="epoch", y="sent_length", data=sent_length_sample_df)
```

```python
from scipy import stats
stats.ttest_ind(
    sent_length_sample_df.query("epoch == 'romantik'").sent_length.to_numpy(),
    sent_length_sample_df.query("epoch == 'realismus'").sent_length.to_numpy(),
    equal_var=False
)
```

```python
round(1.7516366254499303e-153)
```

Hier ist das Ergebniss weniger eindeutig. Auch wenn es so scheint, dass Sätze aus Werken der Romantik tendienziell länger, als solche aus dem Realismus sind.


### Satzbaukomplexität


Als nächster Schritt überprüfen wir mithilfe des `predictive modeling` (Underwood), ob wir Unterschiede in Bezug auf die Satz feststellen können. Hierzu samplen wir wieder Sätze aus beiden Epochen. Anstatt der Tokens verwenden wir jedoch POS-Tags und trainieren ein Klassifikationsverfahren, mithilfe von Uni-, Bi- und Trigrammen der POS-Tags zwischen Sätzen aus der Romantik und dem Realismus zu unterscheiden. Ebenso wie bei den Satzlängen werden die Trainingsdaten ausbalanciert, damit der Classifier nicht durch die Anzahl der Sätze pro Epoche gebiased wird.

```python
df["pos"] = df["tagged_text"].str.findall(r"<.+?>").apply(lambda pos_list: " ".join(pos_list))
```

```python
rom_sents_pos = list(flatten(df.query("epoch == 'romantik'").pos.apply(lambda tags: [sent.strip() for sent in tags.split("<$.>")]).to_list()))
rea_sents_pos = list(flatten(df.query("epoch == 'realismus'").pos.apply(lambda tags: [sent.strip() for sent in tags.split("<$.>")]).to_list()))
```

```python
rom_sents_pos = list(filter(lambda sent: len(sent.split(" ")) > 3, rom_sents_pos))
rea_sents_pos = list(filter(lambda sent: len(sent.split(" ")) > 3, rea_sents_pos))
```

```python
pos_sample_df = pd.DataFrame({
    "sent": rom_sents_pos + rea_sents_pos,
    "epoch": ["romantik"] * len(rom_sents_pos) + ["realismus"] * len(rea_sents_pos)
})
pos_sample_df.epoch.value_counts()
```

```python
pos_train_rom = pos_sample_df.query("epoch == 'romantik'").sample(100_000, random_state=43)
pos_train_rea = pos_sample_df.query("epoch == 'realismus'").sample(100_000, random_state=43)
pos_train_df = pd.concat([pos_train_rom, pos_train_rea]).sample(frac=1.0, random_state=43)
pos_clf_df = pos_sample_df.drop(pos_train_df.index)
pos_test_rom = pos_clf_df.query("epoch == 'romantik'").sample(60_000, random_state=43)
pos_test_rea = pos_clf_df.query("epoch == 'realismus'").sample(60_000, random_state=43)
pos_test_df = pd.concat([pos_test_rom, pos_test_rea]).sample(frac=1.0, random_state=43)
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

countvec = CountVectorizer(ngram_range=(1,3), max_features=500, tokenizer=lambda text: text.split(" "))
#countvec = TfidfVectorizer(use_idf=False, norm="l1", ngram_range=(1,3), max_features=500, tokenizer=lambda text: text.split(" "))
X_train = countvec.fit_transform(pos_train_df.sent)
X_test = countvec.transform(pos_test_df.sent)
```

```python
y_train, y_test = pos_train_df.epoch.to_numpy(), pos_test_df.epoch.to_numpy()
np.unique(y_train, return_counts=True), np.unique(y_test, return_counts=True)
```

```python
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
#logreg = LogisticRegression(C=1, penalty="l1", solver="saga", max_iter=2_000)
logreg = SGDClassifier(loss="log", penalty="l1", alpha=0.0001, max_iter=100_000, n_iter_no_change=100, n_jobs=-1)

logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
```

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_logreg))
```

```python
feats = ["" for _ in range(len(countvec.vocabulary_))]
for feat, idx in countvec.vocabulary_.items():
    feats[idx] = feat
feats = np.array(feats)
feats.shape
```

```python
print(*feats[logreg.coef_.reshape(-1).argsort()][:20], sep="\n")
```

```python
print(*feats[logreg.coef_.reshape(-1).argsort()][-20:][::-1], sep="\n")
```

### Test: Wie funktioniert es mit richtigen Sätzen

```python
from nltk import sent_tokenize
from tqdm.auto import tqdm
def sentenize_dataset(df, text_col="lemmatized_text"):
    data = []
    for _, row in tqdm(df.iterrows()):
        index = row["index"]
        metadata = {
            "orig_index": index,
            "pub_year_estim": row["pub_year_estim"],
            "epoch": row["epoch"],
        }
        for sent in sent_tokenize(row[text_col], language="german"):
            entry = metadata.copy()
            entry[text_col.replace("text", "sent")] = sent
            data.append(entry)
    return pd.DataFrame.from_records(data)
            
```

```python
lemm_sents_df = sentenize_dataset(df)
```

```python
from sklearn.model_selection import train_test_split

lemm_sents_df_train, lemm_sents_df_test = train_test_split(lemm_sents_df, test_size=0.2, random_state=42)
```

```python
lemm_sents_train_rom = lemm_sents_df.query("epoch == 'romantik'").sample(120_000, random_state=43)
lemm_sents_train_rea = lemm_sents_df.query("epoch == 'realismus'").sample(120_000, random_state=43)
lemm_sents_df_train = pd.concat([lemm_sents_train_rom, lemm_sents_train_rea]).sample(frac=1.0, random_state=43)
lemm_sents_clf_df = lemm_sents_df.drop(lemm_sents_df_train.index)
lemm_sents_test_rom = lemm_sents_clf_df.query("epoch == 'romantik'").sample(20_000, random_state=43)
lemm_sents_test_rea = lemm_sents_clf_df.query("epoch == 'realismus'").sample(20_000, random_state=43)
lemm_sents_df_test = pd.concat([lemm_sents_test_rom, lemm_sents_test_rea]).sample(frac=1.0, random_state=43)
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=500)

X_train = tfidf.fit_transform(lemm_sents_df_train.lemmatized_sent)
X_test = tfidf.transform(lemm_sents_df_test.lemmatized_sent)

y_train = lemm_sents_df_train.epoch.to_numpy()
y_test = lemm_sents_df_test.epoch.to_numpy()
```

```python
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report

clf = LogisticRegression(
    C=3,
    solver="saga",
    penalty="l1",
    n_jobs=-1
).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

feats = ["" for _ in range(len(tfidf.vocabulary_))]
for feat, idx in tfidf.vocabulary_.items():
    feats[idx] = feat
feats = np.array(feats)

print(clf.classes_[0], end=":\n\n")
print(*feats[clf.coef_.reshape(-1).argsort()][:20], sep="\n")
print()
print("#"*60,)
print()
print(clf.classes_[1], end=":\n\n")
print(*feats[clf.coef_.reshape(-1).argsort()][-20:][::-1], sep="\n")
```

### Entropy

```python
from scipy.stats import entropy

def sent_entropy(sent):
    tokens = sent.split()
    uniq, counts = np.unique(tokens, return_counts=True)
    return entropy(counts)
```

```python
pos_test_df["sent_ent"] = pos_test_df.sent.apply(sent_entropy)
```

```python
pos_test_df["logreg_pred"] = y_pred_logreg
```

```python
pos_test_df.groupby("epoch")["sent_ent"].mean()
```

```python
sns.histplot(x="sent_ent", hue="epoch", kde=True, data=pos_test_df)
```

```python
sns.boxenplot(x="epoch", y="sent_ent", data=pos_test_df)
```

#### Vergleich: Beschreibung der Individuen


Um zu überprüfen, ob Individuen in den Epochen wirklich anders beschrieben werden, müssen wir eigene Datenerstellen. Hierzu labeln wir 100 Sätze die wir zufällig aus beiden Epochen ziehen, ja nachdem ob sie das in dem Satz beschriebene Individium, 


##
