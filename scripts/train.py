#!/usr/bin/env python3
import argparse, json, os, math
from typing import Dict, List
import random
import numpy as np
import torch.nn.functional as F

# ---------------- Config loader (YAML or JSON) ----------------
def load_config(path: str) -> dict:
    if path.endswith((".yml", ".yaml")):
        try:
            import yaml
        except ImportError:
            raise SystemExit("Please `pip install pyyaml` to use YAML configs.")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- HF/Datasets imports ----------------
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
)
import torch
from dec_hybrid.models_hf import LLTransformerConfig, LLTransformerForCausalLM


# ---------------- Helpers ----------------
def group_texts(examples, block_size: int):
    # Concatenate
    input_ids = sum(examples["input_ids"], [])
    total_len = (len(input_ids) // block_size) * block_size
    input_ids = input_ids[:total_len]
    input_blocks = [input_ids[i:i+block_size] for i in range(0, total_len, block_size)]

    result = {"input_ids": input_blocks, "labels": [blk[:] for blk in input_blocks]}

    if "attention_mask" in examples:
        attn = sum(examples["attention_mask"], [])
        attn = attn[:total_len]
        attn_blocks = [attn[i:i+block_size] for i in range(0, total_len, block_size)]
        # IMPORTANT: ensure same number of blocks as input_ids
        assert len(attn_blocks) == len(input_blocks), (len(attn_blocks), len(input_blocks))
        result["attention_mask"] = attn_blocks

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', required=True, help="Path to YAML or JSON config file.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg.get("model", {})
    data_cfg  = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    log_cfg   = cfg.get("logging", {})
    hub_cfg   = cfg.get("hub", {})
    
    seed = int(train_cfg.get("seed", 420))
    random.seed(seed)
    np.random.seed(seed)
    # ------------- Tokenizer (default: Llama 3; switch to Llama 4 by changing tokenizer_name) -------------
    tokenizer_name = data_cfg.get("tokenizer_name", "meta-llama/Meta-Llama-3-8B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------- Dataset: FineWeb-Edu -------------
    dataset_name  = data_cfg.get("dataset_name", "HuggingFaceFW/fineweb-edu")
    dataset_split = data_cfg.get("split", "train")
    text_column   = data_cfg.get("text_column", "text")
    streaming     = data_cfg.get("streaming", False)

    ds = load_dataset(dataset_name, split=dataset_split, streaming=streaming)

    # Optional small subset for quick tests
    max_train_samples = data_cfg.get("max_train_samples", None)
    if streaming:
        # Convert a limited stream to map-style dataset if limiting samples
        if max_train_samples is not None:
            from itertools import islice
            import pandas as pd, datasets as ds_lib
            rows = list(islice(ds, int(max_train_samples)))
            ds = ds_lib.Dataset.from_pandas(pd.DataFrame(rows))
        else:
            # safety cap to avoid pulling the entire stream inadvertently
            ds = ds.take(10_000).with_format("python")

    # Make a small eval split
    if isinstance(ds, DatasetDict):
        dsd = ds
    else:
        n_total = len(ds)
        eval_ratio = data_cfg.get("eval_ratio", 0.01)
        if n_total and eval_ratio and eval_ratio > 0:
            dsd = ds.train_test_split(test_size=max(1, int(n_total * eval_ratio)))
            dsd = DatasetDict(train=dsd["train"], test=dsd["test"])
        else:
            from datasets import DatasetDict as _DD
            dsd = _DD(train=ds, test=None)

    # Tokenize -> pack into fixed-length blocks
    block_size = int(data_cfg.get("block_size", 1024))

    def tokenize_fn(batch):
        return tokenizer(batch[text_column], truncation=True, add_special_tokens=True)

    # remove_cols = [c for c in [text_column] if c in dsd["train"].column_names]
    # tokenized_train = dsd["train"].map(tokenize_fn, batched=True, remove_columns=remove_cols)
    # tokenized_eval = dsd["test"].map(tokenize_fn, batched=True, remove_columns=remove_cols) if dsd["test"] is not None else None
    tokenized_train = dsd["train"].map(tokenize_fn, batched=True, remove_columns=dsd["train"].column_names)
    tokenized_eval  = dsd["test"].map(tokenize_fn, batched=True, remove_columns=dsd["test"].column_names) if dsd["test"] is not None else None

    # keep only the fields we’ll pack
    keep = [k for k in ["input_ids", "attention_mask"] if k in tokenized_train.column_names]
    tokenized_train = tokenized_train.remove_columns([c for c in tokenized_train.column_names if c not in keep])
    if tokenized_eval is not None:
        tokenized_eval = tokenized_eval.remove_columns([c for c in tokenized_eval.column_names if c not in keep])

    lm_train = tokenized_train.map(lambda e: group_texts(e, block_size), batched=True)
    lm_eval  = tokenized_eval.map(lambda e: group_texts(e, block_size), batched=True) if tokenized_eval is not None else None

    # ------------- Data collator (causal LM) -------------
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ------------- Build config & model via Auto* (thanks to your registration) -------------


    ll_config = LLTransformerConfig(
        vocab_size=model_cfg.get("vocab_size", len(tokenizer)),
        dim=model_cfg.get("dim", 512),
        depth=model_cfg.get("depth", 6),
        num_heads=model_cfg.get("num_heads", 8),
        head_dim=model_cfg.get("head_dim", 64),
        hkv_processor_factory=model_cfg.get("hkv_processor_factory", "decoder-only"),
    )



    model = torch.compile(LLTransformerForCausalLM(ll_config))

    def compute_loss(outputs, labels, num_items_in_batch):
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if num_items_in_batch:
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1), ignore_index=-100, reduction='sum')
        else:
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1), ignore_index=-100, reduction='mean')
                 
        return loss / num_items_in_batch if num_items_in_batch else loss

    # ------------- Weights & Biases -------------
    report_to = train_cfg.get("report_to", ["wandb"])
    if "wandb" in report_to:
        os.environ.setdefault("WANDB_PROJECT", log_cfg.get("wandb_project", "lltransformer"))
        if log_cfg.get("wandb_run_name"):
            os.environ.setdefault("WANDB_NAME", log_cfg["wandb_run_name"])

    # ------------- TrainingArguments -------------
    output_dir = train_cfg.get("output_dir", "./lltransformer-checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=train_cfg.get("learning_rate", 5e-4),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),  # -1 means use epochs
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=train_cfg.get("logging_steps", 50),
        save_steps=train_cfg.get("save_steps", 1000),
        eval_strategy=train_cfg.get("eval_strategy", "steps" if lm_eval is not None else "no"),
        eval_steps=train_cfg.get("eval_steps", 1000),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        bf16=train_cfg.get("bf16", torch.cuda.is_available()),
        fp16=train_cfg.get("fp16", False),
        # gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
        report_to=report_to,
        push_to_hub=hub_cfg.get("push_to_hub", True),
        hub_model_id=hub_cfg.get("hub_model_id", None),
        hub_strategy=hub_cfg.get("hub_strategy", "every_save"),
        hub_private_repo=hub_cfg.get("hub_private_repo", False),
        seed=train_cfg.get("seed", 420),
    )

    # ------------- Trainer -------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_loss_func=compute_loss,
    )

    # ------------- Train/Eval/Save/Push -------------
    train_result = trainer.train()
    trainer.save_model()                # saves to output_dir
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(lm_train)
    if lm_eval is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
        if "eval_loss" in eval_metrics:
            try:
                metrics["eval_perplexity"] = float(math.exp(eval_metrics["eval_loss"]))
            except OverflowError:
                metrics["eval_perplexity"] = float("inf")

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.push_to_hub:
        # Push final checkpoint
        trainer.push_to_hub(commit_message="End of training: push final checkpoint")
        # Push tokenizer to the same repo
        target_repo = hub_cfg.get("hub_model_id", None) or training_args.hub_model_id
        if target_repo:
            tokenizer.push_to_hub(target_repo)

    print("Done.")

if __name__ == "__main__":
    main()
