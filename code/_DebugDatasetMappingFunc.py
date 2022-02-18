#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import Dataset
from transformers import AutoTokenizer


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")


# In[ ]:


from random import shuffle
from random import seed as set_seed
from nltk import sent_tokenize


class SlidingSentShuffle:
    
    def __init__(self,
                 tokenizer,
                 text_column: str = "text",
                 shuffled_text_column: str = "shuffled",
                 target_column: str = "labels",
                 max_length: int = 20,
                 stride: int = 1,
                 language: str = "german",
                 random_state: int = 42
                ):
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.shuffled_text_column = shuffled_text_column
        self.target_column = target_column
        self.max_length = max_length
        self.stride = stride
        self.language = language
        self.random_state = random_state
        set_seed(random_state)
    
    @staticmethod
    def strided_spans(text, window=128, stride=96, tokenizer=lambda text: text.split()):
        tokens = tokenizer(text)
        spans = []
        start = 0
        for i in range(len(tokens) // stride):
            spans.append(tokens[start: start + window])
            if (start + window) >= len(tokens):
                break
            start += stride
        else:
            if start < len(tokens):
                spans.append(tokens[start:])

        #spans = [" ".join(entry) for entry in spans]
        return spans
    
    def sent_tokenize(self, text):
        return sent_tokenize(text, language=self.language)
    
    def shuffle_spans(self, spans):
        shuffled_spans = []
        target_idx = []
        for span in spans:
            orig_idx = list(range(len(span)))
            sents_with_idx = list(zip(span, orig_idx))
            shuffle(sents_with_idx)
            shuffled_span = f"{self.tokenizer.cls_token} " + f" {self.tokenizer.cls_token} ".join(
                [s[0] for s in sents_with_idx]
            )
            so_targets = [s[1] for s in sents_with_idx]
            shuffled_spans.append(shuffled_span)
            target_idx.append(so_targets)
        return shuffled_spans, target_idx
    
    def __call__(self, batch):
        entries_as_dicts = [
            dict(zip(batch, values)) for values in zip(*batch.values())
        ]
        converted_entries = []
        for entry in entries_as_dicts:
            text = entry["text"]
            spans = self.strided_spans(
                text=text,
                window=self.max_length,
                stride=1 if self.stride is None else self.stride,
                tokenizer=self.sent_tokenize
            )
            shuffled_spans, target_idx = self.shuffle_spans(spans)
            for shuffled_span, targets in zip(shuffled_spans, target_idx):
                #shuffled_entry = entry.copy()
                #shuffled_entry.pop(self.text_column)
                shuffled_entry = {}
                shuffled_entry["shuffled"] = shuffled_spans
                shuffled_entry["labels"] = targets
                converted_entries.append(shuffled_entry)
                
        new_entry = {
            key: [entry[key] for entry in converted_entries]
            for key in converted_entries[0]
        }
        return new_entry   


# In[ ]:


sliding_shuffle = SlidingSentShuffle(tokenizer, "text")


# In[ ]:


dataset = Dataset.from_csv("../data/rom_real_dataset_final.csv")
dataset = dataset.remove_columns(['author_pnd', 'author_gender', 'author_city_of_birth', 'author_country_of_birth', 'author_date_of_birth', 'creation_year', 'pub_year', 'pub_place', 'normed_text', 'tagged_text', 'lemmatized_text'])
dataset


# In[ ]:


shuffled_dataset = dataset.map(
    sliding_shuffle,
    batched=True,
    batch_size=1,
    remove_columns=[sliding_shuffle.text_column],
)


# In[ ]:


test = " ".join(f"Das ist Satz {i}." for i in range(100))


# In[ ]:


sents = sliding_shuffle.sent_tokenize(test)


# In[ ]:


spans = sliding_shuffle.strided_spans(test, window=2, stride=1, tokenizer=sliding_shuffle.sent_tokenize)


# In[ ]:


sliding_shuffle.shuffle_spans(spans)


# In[ ]:




