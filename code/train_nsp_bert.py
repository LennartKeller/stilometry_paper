from random import choice, seed

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from more_itertools import windowed
from nltk import sent_tokenize
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EvalPrediction, Trainer, TrainingArguments)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

def make_nsp_dataset(df, random_state=42):
    seed(random_state)
    data = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Creating NSP dataset"):
        text = row.pop("text")
        sents = sent_tokenize(text, language="german")
        sent_idc = list(range(len(sents)))
        for index, pair in enumerate(windowed(sents, 2)):
            sent_1_idx, sent_2_idx = index, index + 1
            sent_1, sent_2 = pair 
            
            true_pair = " ".join(pair)
            true_entry = row.copy()
            true_entry["pair"] = true_pair
            true_entry["sent1_idx"] = sent_1_idx
            true_entry["sent2_idx"] = sent_2_idx
            true_entry["sent1"] = sent_1
            true_entry["sent2"] = sent_2
            true_entry["label"] = 0
            data.append(true_entry)
            
            # chose any sent from same text that is not sent_2
            while True:
                false_sent_idx = choice(sent_idc)
                false_sent = sents[false_sent_idx]
                if sent_2 != false_sent:
                    break
            
            false_pair = " ".join((sent_1, false_sent))
            false_entry = row.copy()
            false_entry["pair"] = false_pair
            false_entry["sent1_idx"] = sent_1_idx
            false_entry["sent2_idx"] = false_sent_idx
            false_entry["sent1"] = sent_1
            false_entry["sent2"] = false_sent
            false_entry["label"] = 1
            data.append(false_entry)

    return pd.DataFrame.from_records(data)    


if __name__ == "__main__":
    # Preprocessing
    corpus = pd.read_csv("../data/rom_real_dataset_final.csv")
    corpus = corpus[["author", "title", "epoch", "text"]]
    corpus["text"] = corpus.text.str.replace("[»›]", '"', regex=True)
    corpus["text"] = corpus.text.str.replace("[«‹]", '"', regex=True)
    corpus["text"] = corpus.text.str.replace("–", '-', regex=True)
    corpus.text.str.contains("[»›]").any(), corpus.text.str.contains("[«‹]").any(), corpus.text.str.contains("–").any()

    # Splitting
    corpus_train = pd.concat([
        corpus.query("epoch == 'romantik'").sample(5, random_state=42),
        corpus.query("epoch == 'realismus'").sample(5, random_state=42)
    ])
    corpus_test = corpus.drop(corpus_train.index)

    with open("../data/nsp_corpus_train_idx.txt", "w") as f:
        f.write("\n".join(map(str, corpus_train.index)))

    with open("../data/nsp_corpus_test_idx.txt", "w") as f:
        f.write("\n".join(map(str, corpus_test.index)))

    # Datset creation
    train_dataset = make_nsp_dataset(corpus_train)
    test_dataset = make_nsp_dataset(corpus_test)

    hf_dataset = DatasetDict({
        "train": Dataset.from_pandas(train_dataset),
        "test": Dataset.from_pandas(test_dataset)
    })
    hf_dataset.save_to_disk("../data/rom_rea_nsp_hf_dataset")

    # Train preperation
    model = AutoModelForSequenceClassification.from_pretrained("bert-nsp", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

    hf_dataset = hf_dataset.map(
        lambda entry: tokenizer(entry["pair"], padding="max_length", truncation=True),
        batched=True,
        num_proc=12
    )

    hf_dataset = hf_dataset.rename_column("label", "labels")
    hf_dataset.set_format("torch")

    hf_train_dataset = hf_dataset["train"]
    hf_train_dataset = hf_train_dataset.train_test_split(train_size=0.98)
    hf_train_dataset

    training_args = TrainingArguments(
        num_train_epochs=3,
        output_dir="so_classif",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=48,
        learning_rate=3e-5,
        logging_dir="so_classif/logs",
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        evaluation_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=5,
        overwrite_output_dir=True,
        warmup_steps=350,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=hf_train_dataset["train"],
        eval_dataset=hf_train_dataset["test"],
        compute_metrics=compute_metrics
    )

    # Training
    trainer.train()
    model.save_pretrained("bert-nsp")
    tokenizer.save_pretrained("bert-nsp")
    
    # Prediction
    predictions = trainer.predict(hf_dataset["test"])
    test_dataset = hf_dataset["test"].to_pandas()
    test_dataset["pred"] = predictions.predictions.argmax(axis=1)
    test_dataset["match"] = (test_dataset["labels"] == test_dataset["pred"]).astype("int")
    test_dataset = test_dataset.drop(
        columns=["input_ids", "token_type_ids", "attention_mask"]
    )
    test_dataset.to_csv("../data/testdataset_bert.csv", index=False)
