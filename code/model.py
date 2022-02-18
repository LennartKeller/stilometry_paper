import re
from collections import defaultdict
from typing import Callable, Dict, List
from dataclasses import dataclass, field
from string import punctuation

import numpy as np
import torch
from datasets import Dataset
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score
from torch import nn
from torch._C import EnumType
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    EvalPrediction,
    PreTrainedTokenizer,
    Trainer,
    default_data_collator,
    DataCollatorWithPadding
)


@dataclass
class ModelArgs:
    
    dataset_path: str = field(
        metadata={
            "help": "Path to the dataset."
        },
    )

    validation_split: float = field(
        default=0.1,
        metadata={
            "help": "Percentage of train data to use for validation."
        },
    )

    model_name_or_path: str = field(
        default="bert-base-cased",
        metadata={
            "help": "Path to pretrained model or model or its name to load it from Huggingface Hub."
        },
    )

    final_checkpoint_path: str = field(
        default=None, metadata={"help": "Path to save the final model."}
    )

    normalize_targets: bool = field(
        default=False,
        metadata={
            "help": "Normalize regression targets to be in range [0,1] | Default False"
        }
    )

    window: int = field(
        default=None,
        metadata={
            "help": "Size of the sliding window which is used to generate train instances"
        }
    )

    padding: int = field(
        default=None,
        metadata={
            "help": "Size of the context windows (left and right) which are only used to give context the model."
        }
    )
    
    stride: int = field(
        default=None,
        metadata={
            "help": "Number of tokens the sliding window is moved to the right while generating train instances"
        }
    )


def make_compute_metrics_func(target_token_id) -> Callable:
    def compute_ranking_func(eval_prediction: EvalPrediction) -> Dict[str, float]:
        batch_sent_idx, batch_input_ids = eval_prediction.label_ids
        # We convert the logits with shape (batch_size, seq_len, 1) to be in shape (batch_size, seq_len)
        batch_logits = eval_prediction.predictions.squeeze(2)

        metrics = defaultdict(list)
        for sent_idx, input_ids, logits in zip(
            batch_sent_idx, batch_input_ids, batch_logits
        ):
            sent_idx = sent_idx.reshape(-1)
            input_ids = input_ids.reshape(-1)
            logits = logits.reshape(-1)
            # Custom: NORM 
            sent_idx = np.argsort(sent_idx[sent_idx != -100])
            target_logits = logits[input_ids == target_token_id]
            if sent_idx.shape[0] > target_logits.shape[0]:
                sent_idx = sent_idx[: target_logits.shape[0]]
                sent_idx = np.argsort(sent_idx)
            # Calling argsort twice on the logits gives us their ranking in ascending order
            predicted_idx = np.argsort(np.argsort(target_logits))
            tau, pvalue = kendalltau(sent_idx, predicted_idx)
            if not np.isnan(tau):
                metrics["kendalls_tau"].append(tau)
            metrics["acc"].append(accuracy_score(sent_idx, predicted_idx))
            metrics["mean_logits"].append(logits.mean())
            metrics["std_logits"].append(logits.std())
        metrics = {metric: np.mean(scores) for metric, scores in metrics.items()}
        return metrics

    return compute_ranking_func

class SoDataCollator:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
            )
    
    def __call__(self, batch_entries):
        label_dicts = []
        # We split the labels from the rest to process them independently
        for entry in batch_entries:
            label_dict = {}
            for key in list(entry.keys()):
                if "labels" in key:
                    label_dict[key] = entry.pop(key)
            label_dicts.append(label_dict)
        # Everything except our labels can easily be handled be transformers default collator
        batch = self.collator(batch_entries)
        #batch = default_data_collator(batch_entries)

        # We need to pad the labels "manually"
        for label in label_dicts[0]:
            labels = pad_sequence(
                [label_dict[label] for label_dict in label_dicts],
                batch_first=True,
                padding_value=-100,
            )

            batch[label] = labels
        return batch

        
def so_data_collator(batch_entries):
    """
    Custom dataloader to apply padding to the labels.
    TODO document me better :)
    """
    label_dicts = []

    # We split the labels from the rest to process them independently
    for entry in batch_entries:
        label_dict = {}
        for key in list(entry.keys()):
            if "labels" in key:
                label_dict[key] = entry.pop(key)
        label_dicts.append(label_dict)
    print([b.keys() for b in batch_entries])
    # Everything except our labels can easily be handled be transformers default collator
    batch = DataCollatorWithPadding(tokenizer=tokenizer)(batch_entries)
    #batch = default_data_collator(batch_entries)

    # We need to pad the labels "manually"
    for label in label_dicts[0]:
        labels = pad_sequence(
            [label_dict[label] for label_dict in label_dicts],
            batch_first=True,
            padding_value=-100,
        )

        batch[label] = labels

    return batch


class SentenceOrderingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.target_token_id = kwargs.pop("target_token_id")
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        # Get sentence indices
        batch_labels = inputs.pop("labels")
        # Get logits from model
        outputs = model(**inputs)
        batch_logits = outputs["logits"]

        # Get logits for all cls tokens
        batch_input_ids = inputs["input_ids"]

        # Since we have varying number of labels per instance, we need to compute the loss manually for each one.
        loss_fn = nn.MSELoss(reduction="sum")
        batch_loss = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

        for labels, logits, input_ids in zip(
            batch_labels, batch_logits, batch_input_ids
        ):

            # Firstly, we need to convert the sentence indices to regression targets.
            # To avoid exploding gradients, we norm them to be in range 0 <-> 1
            # Also we need to remove the padding entries (-100)
            true_labels = labels[labels != -100].reshape(-1)
            targets = true_labels.float()

            # Secondly, we need to get the logits from each target token in the input sequence
            target_logits = logits[input_ids == self.target_token_id].reshape(-1)
            # Sometimes we will have less target_logits than targets due to trunction of the input
            # In this case, we just consider as many targets as we have logits
            if target_logits.size(0) < targets.size(0):
                targets = targets[: target_logits.size(0)]

            # Finally we compute the loss for the current instance and add it to the batch loss
            batch_loss = batch_loss + loss_fn(targets, target_logits)

        # The final loss is obtained by averaging over the number of instances per batch
        loss = batch_loss / batch_logits.size(0)

        outputs["loss"] = loss
        return (loss, outputs) if return_outputs else loss


def make_tokenization_func(tokenizer, text_column, *args, **kwargs):
    def tokenization(entry):
        return tokenizer(entry[text_column], *args, **kwargs)

    return tokenization


def make_rename_func(mapping, remove_src=False):
    def rename(entry):
        for src, dst in mapping.items():
            if remove_src:
                data = entry.pop(src)
            else:
                data = entry[src]
            entry[dst] = data
        return entry

    return rename

import pandas as pd
from random import shuffle
from random import seed as set_seed
from nltk import sent_tokenize
from tqdm.auto import tqdm
from copy import deepcopy
import numpy as np

class SlidingSentShuffle:
    
    def __init__(self,
                 tokenizer,
                 text_column: str = "text",
                 shuffled_text_column: str = "shuffled",
                 target_column: str = "labels",
                 max_length: int = 10,
                 stride: int = 8,
                 norm_targets: bool = False,
                 language: str = "german",
                 random_state: int = 42
                ):
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.shuffled_text_column = shuffled_text_column
        self.target_column = target_column
        self.max_length = max_length
        self.stride = stride
        self.norm_targets = norm_targets
        self.language = language
        self.random_state = random_state
        set_seed(random_state)
    
    @staticmethod
    def strided_spans(text, window, stride, tokenizer=lambda text: text.split()):
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
            # Custom: NORM
            so_targets = np.array([s[1] for s in sents_with_idx])
            if self.norm_targets:
                so_targets = np.array([s[1] for s in sents_with_idx]) / so_targets.max()
            shuffled_spans.append(shuffled_span)
            target_idx.append(so_targets)
        return shuffled_spans, target_idx
    
    def convert_dataframe(self, df: pd.DataFrame):
        converted_entries = []
        for index, row in tqdm(df.iterrows()):
            text = row["text"]
            spans = self.strided_spans(
                text=text,
                window=self.max_length,
                stride=1 if self.stride is None else self.stride,
                tokenizer=self.sent_tokenize
            )
            shuffled_spans, target_idx = self.shuffle_spans(spans)
            for span_idx, (shuffled_span, targets) in enumerate(zip(shuffled_spans, target_idx)):
                shuffled_entry = row.copy()
                shuffled_entry.pop(self.text_column)
                shuffled_entry["span_idx"] = span_idx
                shuffled_entry["shuffled"] = shuffled_span
                shuffled_entry["labels"] = targets
                converted_entries.append(shuffled_entry)
            
        return pd.DataFrame.from_records(converted_entries)
    
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
            for span_idx, (shuffled_span, targets) in enumerate(zip(shuffled_spans, target_idx)):
                shuffled_entry = entry.copy()
                shuffled_entry.pop(self.text_column)
                shuffled_entry["span_idx"] = span_idx
                shuffled_entry["shuffled"] = shuffled_span
                shuffled_entry["labels"] = targets
                converted_entries.append(shuffled_entry)
                
        new_entry = {
            key: [entry[key] for entry in converted_entries]
            for key in converted_entries[0]
        }
        return new_entry
    

class SlidingTokenShuffle:

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_column: str = "text",
        target_column: str = "labels",
        window: int = None,
        padding: int = None,
        stride: int = None,
        norm_targets: bool = False,
        language: str = "german",
        random_state: int = 42
        ):
        
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.target_column = target_column
        self.window = tokenizer.model_max_length if window is None else window
        self.padding = self.window // 4 if padding is None else padding
        self.stride = self.padding if stride is None else stride
        self.norm_targets = norm_targets
        self.language = language
        self.random_state = 42
        set_seed(random_state)
    
    def sent_tokenize(self, text: str) -> List[str]:
        return sent_tokenize(text, language=self.language)
    
    @staticmethod
    def normalize_whitespace(string: str) -> str:
        return " ".join(string.split())
    

    def __call__(self, batch):
        entries_as_dicts = [
            dict(zip(batch, values)) for values in zip(*batch.values())
        ]
        converted_entries = []
        for entry in entries_as_dicts:
            text = entry[self.text_column]
            sents = self.sent_tokenize(text)
            prepared_text = f"{self.tokenizer.cls_token} " + f" {self.tokenizer.cls_token} ".join(sents)
            tokenized = self.tokenizer(
                prepared_text,
                add_special_tokens=False,
                max_length=None,
                verbose=False,
                return_tensors="pt"
                )
            # indices of the tokenized sequence
            seq_idx = torch.arange(tokenized["input_ids"].reshape(-1).size(0))
            if seq_idx.size(0) > self.window:
                window_idc = seq_idx.unfold(0, self.window, self.stride)
            else:
                # If the text is shorter than the window length we do not need to create windows...
                window_idc = seq_idx.unsqueeze(0)

            window_index = 0
            for window_idx in window_idc:
                # Create new "input_dict"
                window_tokenized = {
                    key: tokenized[key][0, window_idx]
                    for key in tokenized
                }
                window_input_ids = window_tokenized["input_ids"]
                
                # Shuffle text in the middle of the window
                if self.padding > 0:
                    middle_tokens = window_input_ids[self.padding:-self.padding]
                else:
                    middle_tokens = window_input_ids
                middle_text = self.tokenizer.decode(middle_tokens)
                middle_text = self.normalize_whitespace(middle_text)
                middle_sents = middle_text.split(self.tokenizer.cls_token)
                middle_sents = [
                    sent
                    for sent in middle_sents
                    if sent.strip().strip(punctuation)
                ]
                if not middle_sents:
                    continue
                # we remove the last sent cause its very likely cropped and just add to its predecesor...
                first_sent = ""
                last_sent = ""
                if len(middle_sents) >= 4:
                    first_sent = middle_sents.pop(0)
                    last_sent = middle_sents.pop(-1)
                n_shuffled_sents = len(middle_sents)
                middle_sents_idx = list(range(len(middle_sents)))
                middle_sents_with_idx = list(zip(middle_sents, middle_sents_idx))
                shuffle(middle_sents_with_idx)
                shuffled_sents = [i[0] for i in middle_sents_with_idx]
                so_targets = [i[1] for i in middle_sents_with_idx]
                if self.norm_targets:
                    so_targets = [i / max(max(so_targets),1) for i in so_targets]
                shuffled_text = f"{self.tokenizer.cls_token} " + f" {self.tokenizer.cls_token} ".join(shuffled_sents)
                
                # Remove all cls tokens from the padding regions of the text
                if self.padding > 0:
                    padding_tokens_left = window_input_ids[:self.padding]
                else:
                    padding_tokens_left = torch.tensor([])
                padding_text_left = self.tokenizer.decode(padding_tokens_left)
                padding_text_left = padding_text_left.rstrip("##") + first_sent.lstrip("##")
                padding_text_left_converted = self.normalize_whitespace(
                    padding_text_left).replace(self.tokenizer.cls_token, "")
                
                if self.padding > 0:
                    padding_tokens_right = window_input_ids[-self.padding:]
                else:
                    padding_tokens_right = torch.tensor([])
                padding_text_right = self.tokenizer.decode(padding_tokens_right)
                padding_text_right = last_sent.rstrip("##") + padding_text_right.lstrip("##")
                padding_text_right_converted = self.normalize_whitespace(
                    padding_text_right).replace(self.tokenizer.cls_token, "")

                full_text = f"{padding_text_left_converted} {shuffled_text} {padding_text_right_converted}"
                full_text = self.normalize_whitespace(full_text)

                final_tokenized = self.tokenizer(
                    full_text,
                    add_special_tokens=False,
                    max_length=None,
                    verbose=False,
                    return_tensors="pt",

                )
                final_tokenized = {
                    key: final_tokenized[key].reshape(-1)[:self.window].tolist()
                    for key in final_tokenized
                }

                final_tokenized["text"] = full_text
                final_tokenized[self.target_column] = so_targets
                final_tokenized["window_index"] = window_index
                window_index += 1
                final_tokenized["n_shuffled_sents"] = n_shuffled_sents
                
                if len(so_targets) != final_tokenized["text"].count(self.tokenizer.cls_token):
                    raise Exception("Number of targets does not match number of cls_tokens")

                for key in entry:
                    if not key in final_tokenized:
                        final_tokenized[key] = entry[key]
                
                converted_entries.append(final_tokenized)
        
        new_entry = {
            key: [entry[key] for entry in converted_entries]
            for key in converted_entries[0]
        }
        return new_entry


import torch
import numpy as np
from typing import List
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedModel
from nltk import sent_tokenize
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding


class SentEmbeddingExtractor:
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sent_step_size: int = 1,
        batch_size: int = 16,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.sent_step_size = sent_step_size
        self.batch_size = batch_size
        self.device = device
    
    def prepare_text(self, text: str) -> List[BatchEncoding]:
        sents = self.sent_tokenize(text)
        encodings = []
        current_sents = []
        pbar = tqdm(total=len(sents), desc="Chunking sentences")
        while sents:
            next_sent = sents.pop(0)
            pbar.update(1)
            current_sents_n_tokens = self.tokenizer(
                self._join_sents(current_sents),
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"].size(1)
            next_sent_n_tokens = self.tokenizer(
                self._join_sents([next_sent]),
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"].size(1)
            
            if current_sents_n_tokens + next_sent_n_tokens <= self.tokenizer.model_max_length:
                current_sents.append(next_sent)
            else:
                final_inputs = self.tokenizer(
                self._join_sents(current_sents),
                add_special_tokens=False,
                return_tensors="pt"
                )
                final_inputs = BatchEncoding({
                    key: value.reshape(-1)
                    for key, value in final_inputs.items()
                })
                encodings.append(final_inputs)
                current_sents = [next_sent]
            
            if not sents:
                    final_inputs = self.tokenizer(
                    self._join_sents(current_sents),
                    add_special_tokens=False,
                    return_tensors="pt"
                    )
                    final_inputs = BatchEncoding({
                        key: value.reshape(-1)
                        for key, value in final_inputs.items()
                    })
                    encodings.append(final_inputs)
                
                
                    
        return encodings
    
    def _extract_sent_embeddings(self, text_encodings: List[BatchEncoding]) -> np.ndarray:
        batches = self._prepare_batches(text_encodings)
        embeddings = []
        pbar = tqdm(batches, desc="Extracting sentence embeddings")
        for batch in pbar:
            batch = batch.to(self.device)
            with torch.no_grad():
                outputs = self.model(**batch)
                last_hidden_state = outputs["last_hidden_state"]
                input_ids = batch["input_ids"]
                for i, l in zip(input_ids, last_hidden_state):
                    sent_embeddings = l[i == self.tokenizer.cls_token_id].detach().cpu().numpy()
                    embeddings.extend(sent_embeddings)
        return np.array(embeddings)
    
    @staticmethod
    def sent_tokenize(text):
        sents = sent_tokenize(text, language="german")
        return sents
    
    def _join_sents(self, sents: List[str]) -> str:
        return f" {self.tokenizer.cls_token}" + f" {self.tokenizer.cls_token} ".join(sents)
    
    def _prepare_batches(self, text_encodings: List[BatchEncoding]) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            text_encodings,
            batch_size=self.batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True
            )
        )
    
    def extract(self, text: str) -> np.ndarray:
        text_encodings = self.prepare_text(text)
        embeddings = self._extract_sent_embeddings(text_encodings)
        return embeddings

    

   
