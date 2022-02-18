import json
from transformers import TrainingArguments, HfArgumentParser
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer
from transformers import set_seed
from datasets import load_from_disk

from model import (
    SentenceOrderingTrainer,
    so_data_collator,
    make_compute_metrics_func,
    ModelArgs,
    make_tokenization_func,
    SoDataCollator,
    SlidingSentShuffle
)


if __name__ == "__main__":

    args_parser = HfArgumentParser((ModelArgs, TrainingArguments))
    model_args, training_args = args_parser.parse_args_into_dataclasses()

    # Add fixed args
    training_args.label_names = ["labels", "input_ids"]

    set_seed(training_args.seed)

    dataset = load_from_disk(
        model_args.dataset_path
    )
    dataset = dataset["train"]
    

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    if not tokenizer.pad_token:
        print("Tokenizer has no pad_token.. Adding one!")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if not tokenizer.cls_token:
        print("Tokenizer has no cls_token.. Adding one!")
        tokenizer.add_special_tokens({'pad_token': '[CLS]'})

    print("Created shuffled spans from src text...")
    shuffle_func = SlidingSentShuffle(
        tokenizer=tokenizer,
        text_column="text",
        max_length=25,
        stride=5,
        norm_targets=True
        )
    dataset = dataset.map(
        shuffle_func,
        batched=True,
        batch_size=10,
        num_proc=12,
        remove_columns=[shuffle_func.text_column]
        )
    print(f"Created {len(dataset)} shuffled spans!")

    print(f"Setting aside {model_args.validation_split} of samples for validation...")
    dataset = dataset.train_test_split(test_size=model_args.validation_split, seed=training_args.seed)

    print("Loading model")
    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, num_labels=1
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path, config=model_config
    )

    print("Tokenizing texts...")
    tokenization = make_tokenization_func(
        tokenizer=tokenizer,
        text_column="shuffled",
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
        max_length=512
    )
    dataset = dataset.map(tokenization, batched=True, num_proc=12)

    dataset.set_format("torch")
    print("Initialize metrics func")
    metrics_func = make_compute_metrics_func(tokenizer.cls_token_id)

    trainer = SentenceOrderingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        target_token_id=tokenizer.cls_token_id,
        data_collator=SoDataCollator(tokenizer=tokenizer),
        compute_metrics=metrics_func,
    )
    print("Start training...")
    trainer.train()

    trainer.save_model(model_args.final_checkpoint_path)

    test_results = trainer.evaluate(eval_dataset=dataset["test"])
    with open(f"test_results_{model_args.model_name_or_path}.json", "w") as f:
        json.dump(test_results, f)

    print(test_results)
